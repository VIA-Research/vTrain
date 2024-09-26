import torch

from .model.gpt_model import ShardedGptModel

from .trainer import Trainer
from .config import vTrainConfig
from .graph import CommNode, DepGraph, LayerNode, TaskNode

import os
import logging


logger = logging.getLogger()
logging.basicConfig(
    format="[%(asctime)s] (%(levelname)s) %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO
)


model_names = ['bert', 'gpt']


class ParamInfo():
    def __init__(self, elem_num, elem_size=2):
        self.elem_num = elem_num
        self.elem_size = elem_size

    def numel(self):
        return self.elem_num
    
    def element_size(self):
        return self.elem_size
    
    def __repr__(self):
        return f"parameter having {self.elem_num} of {self.elem_size} bytes"


class vTrain():
    def __init__(self, config: vTrainConfig):
        self.config = config

        self.model = None
        self.model_params = {
                                'embeddings': [
                                    ParamInfo((config.vocab_size // config.tensor_parallel_size) * config.hidden_size),        # word embed
                                    ParamInfo(config.max_length * config.hidden_size),                                         # pos embed
                                ],
                                'transformer': [
                                    ParamInfo(config.hidden_size),                                                             # layernorm
                                    ParamInfo(config.hidden_size),                                                             # layernorm
                                    ParamInfo(3 * config.hidden_size * (config.hidden_size // config.tensor_parallel_size)),   # qkv
                                    ParamInfo(3 * config.hidden_size // config.tensor_parallel_size),                          # qkv bias
                                    ParamInfo(config.hidden_size * (config.hidden_size // config.tensor_parallel_size)),       # output proj
                                    ParamInfo(config.hidden_size),                                                             # output proj bias
                                    ParamInfo(config.hidden_size),                                                             # layernorm
                                    ParamInfo(config.hidden_size),                                                             # layernorm
                                    ParamInfo(config.hidden_size * (4 * config.hidden_size // config.tensor_parallel_size)),   # up proj
                                    ParamInfo(4 * config.hidden_size // config.tensor_parallel_size),                          # up proj bias
                                    ParamInfo(4 * config.hidden_size * (config.hidden_size // config.tensor_parallel_size)),   # down proj
                                    ParamInfo(config.hidden_size),                                                             # down proj bias
                                ],
                                'logit': [
                                    ParamInfo((config.vocab_size // config.tensor_parallel_size) * config.hidden_size)         # logit
                                ],
                            }
        self.layers = [('embeddings', True)] + \
                        [('transformer', True) for _ in range(config.num_layers)] + \
                        [('logit', True)]

        self.cbid_table = None
        self.allreduce_LUT = self.get_allreduce_LUT()

    def __call__(self):
        config = self.config

        # create model
        # self.create_model()
        self.graph = DepGraph()

        # logger.info(f"dp, tp, pp = {config.data_parallel_size}, {config.tensor_parallel_size}, {config.pipeline_parallel_size}")
        logger.info(config)

        # create layer graph which contains
        # framework-level information
        logger.info(f"create graph...")
        ingredients = self.create_nodes()
        self.create_layer_graph(ingredients)

        # collect CUDA runtime and GPU kernel traces
        # and replace layer nodes to low-level tasks
        logger.info(f"start profiling...")
        kernel_dict = self.profile()

        # Predict iteration time 
        logger.info(f"start prediction...")
        result, breakdown = self.predict(kernel_dict)

        return result, breakdown
    

    def show_graph(self):
        if self.graph is None:
            logger.error(f"there is no simulated execution graph")
        else:
            self.graph.show_graph()


    def create_model(self):
        config: vTrainConfig = self.config

        torch.set_default_dtype(torch.float16)
        self.model = ShardedGptModel(num_layers=1,
                                     hidden_size=config.hidden_size,
                                     world_size=config.tensor_parallel_size,
                                     num_attention_heads=config.num_attention_heads,
                                     max_sequence_length=config.max_length)

        return True


    def create_nodes(self):
        layers = self.layers
        ingredients = {"fwd": {}, "bwd": {}, "wu": {}}

        # forward functions
        for layer_num, (layer_name, _) in enumerate(layers):
            fwdNode = (layer_num, layer_name, f"Fwd_{layer_name}", "GPU0")
            ingredients["fwd"][layer_num] = fwdNode

        lossNode = (len(layers), "loss", "Fwd_loss", "GPU0")
        ingredients["fwd"][len(layers)] = lossNode

        # backward functions
        for layer_num, (layer_name, requires_grad) in enumerate(layers):
            if not requires_grad:
                continue
            bwdNode = (layer_num, layer_name, f"Bwd_{layer_name}", "GPU0")
            ingredients["bwd"][layer_num] = bwdNode

        # weight update functions
        for layer_num, (layer_name, requires_grad) in enumerate(layers):
            if not requires_grad:
                continue
            WUNode = (layer_num, layer_name, f"WU_{layer_name}", "GPU0")
            ingredients["wu"][layer_num] = WUNode
        
        return ingredients
    

    def _compute_p2p_latency(self,
                             data_size,     # data_size in Bytes
                             bandwidth):    # bandwidth in Gbps
        # simple latency-bandwidth model
        bytes_per_sec = bandwidth * (2 ** 30) / 8
        latency_sec = data_size / (bytes_per_sec / 4)
        latency_nano_sec = latency_sec * (10 ** 9)
        return latency_nano_sec
    

    def create_layer_graph(self, ingredients):
        config = self.config
        graph = self.graph
        nodes_by_layer = [{"fwd": None, "bwd": None, "wu": None}
                            for _ in range(len(self.layers)+1)]
        graph.create_stream("Comm")

        dp, tp, pp = config.data_parallel_size, config.tensor_parallel_size, config.pipeline_parallel_size

        # create streams
        for gpu_num in range(pp):
            graph.create_stream(f"GPU{gpu_num}")
        graph.create_stream("Comm")

        # balacne
        balance = [config.num_layers // pp for _ in range(pp)]
        balance[0] += 1     # embedding
        balance[-1] += 1    # logit
        num_layers = sum(balance)

        layer_idx_by_rank = []
        idx = 0
        for n in balance:
            idx_list = list(range(idx, idx+n))
            layer_idx_by_rank.append(idx_list)
            idx = idx_list[-1] + 1

        num_microbatch = (config.global_batch_size // dp) // config.micro_batch_size

        self.nodes_by_microbatch = [[] for _ in range(num_microbatch)]
        nodes_by_microbatch = self.nodes_by_microbatch
        
        local_batch_size = config.micro_batch_size
        data_size = 2   # bytes
        feature_map_size = local_batch_size * config.max_length * config.hidden_size * data_size

        # warmup phase
        num_warmup_microbatch_rank0 = min(pp - 1, num_microbatch)
        for microbatch_idx in range(num_warmup_microbatch_rank0):
            for rank in range(pp - 1 - microbatch_idx):
                for layer_idx in layer_idx_by_rank[rank]:
                    nodeInfo = ingredients["fwd"][layer_idx]
                    node = LayerNode(*nodeInfo)
                    node.stream = f"GPU{rank}"

                    graph.add_node(node)
                    nodes_by_microbatch[microbatch_idx].append(node)

                    # comm across mp
                    if tp > 1 and nodeInfo[1] in ["encoder", "transformer"]:
                        self._add_tp_communication(rank, tp, microbatch_idx, feature_map_size)
                        self._add_tp_communication(rank, tp, microbatch_idx, feature_map_size)
                    
                # pipeline inter-stream gap
                pp_gap = self._compute_p2p_latency(2*feature_map_size, config.inter_node_bandwidth)
                graph.streams[f"GPU{rank}"][-1].gap += pp_gap

        # 1F1B phase + cooldown phase
        for microbatch_idx in range(num_microbatch):
            for rank in range(pp-1, -1, -1):
                fwd_microbatch_idx = pp - rank - 1 + microbatch_idx
                bwd_microbatch_idx = microbatch_idx

                # Fwd nodes
                if fwd_microbatch_idx < num_microbatch:
                    for layer_idx in layer_idx_by_rank[rank]:
                        nodeInfo = ingredients["fwd"][layer_idx]
                        node = LayerNode(*nodeInfo)
                        node.stream = f"GPU{rank}"

                        graph.add_node(node)
                        nodes_by_microbatch[fwd_microbatch_idx].append(node)

                        # comm across mp
                        if tp > 1 and nodeInfo[1] in ["encoder", "transformer"]:
                            self._add_tp_communication(rank, tp, fwd_microbatch_idx, feature_map_size)
                            self._add_tp_communication(rank, tp, fwd_microbatch_idx, feature_map_size)

                    # comm across pp
                    if rank < pp - 1:
                        pp_gap = self._compute_p2p_latency(2*feature_map_size, config.inter_node_bandwidth)
                        graph.streams[f"GPU{rank}"][-1].gap += pp_gap
                        
                    # loss node
                    if rank == pp - 1:
                        nodeInfo = ingredients["fwd"][num_layers]
                        node = LayerNode(*nodeInfo)
                        node.stream = f"GPU{rank}"

                        graph.add_node(node)
                        nodes_by_microbatch[fwd_microbatch_idx].append(node)

                # Bwd nodes + recomputation
                if config.use_checkpoint:
                    for layer_idx in reversed(layer_idx_by_rank[rank]):
                        if rank < pp - 1:   # Last rank worker dosen't perform recomputation
                            recompNodeInfo = ingredients["fwd"][layer_idx]
                            recompNode = LayerNode(*recompNodeInfo)
                            recompNode.stream = f"GPU{rank}"
                            graph.add_node(recompNode)

                for layer_idx in reversed(layer_idx_by_rank[rank]):
                    nodeInfo = ingredients["bwd"][layer_idx]
                    node = LayerNode(*nodeInfo)
                    node.stream = f"GPU{rank}"

                    graph.add_node(node)
                    nodes_by_microbatch[bwd_microbatch_idx].append(node)

                    # comm across mp
                    if tp > 1 and nodeInfo[1] in ["encoder", "transformer"]:
                        self._add_tp_communication(rank, tp, bwd_microbatch_idx, feature_map_size)
                        self._add_tp_communication(rank, tp, bwd_microbatch_idx, feature_map_size)
                    
                if rank > 0:
                    pp_gap = self._compute_p2p_latency(2*feature_map_size, config.inter_node_bandwidth)
                    graph.streams[f"GPU{rank}"][-1].gap += pp_gap
        
        param_size_by_rank = [0 for _ in range(pp)]
        for rank, layer_nums in enumerate(layer_idx_by_rank):
            size = 0
            for layer_num in layer_nums:
                layer_name, _ = self.layers[layer_num]
                for p in self.model_params[layer_name]:
                    size += p.numel() * p.element_size()
            param_size_by_rank[rank] = size

        # comm across dp
        if dp > 1:
            if pp > 1:
                last_bwd = [self.graph.streams["GPU0"][-1]]
            else:
                last_bwd = []

            for rank in range(pp):
                comm_node = CommNode(param_size_by_rank[rank], "Comm")
                if tp < config.node_size:  # intra-node grad allreduce for dp
                    comm_node.duration = self.compute_comm_time(comm_node.bucket_size, dp)
                else:
                    comm_node.duration = comm_node.bucket_size / (config.inter_node_bandwidth * (2 ** 30) / 8) \
                                            * (2*(dp-1)/dp) * (10 ** 9)
                self.graph.add_node(comm_node, prev=last_bwd)
                self.graph.append_node_to_stream(comm_node, f"GPU{rank}")

        # optimizer step
        last_nodes = [nodes[-1] for nodes in graph.streams.values() if nodes]
        for rank in range(pp):
            for layer_idx in layer_idx_by_rank[rank]:
                nodeInfo = ingredients["wu"][layer_idx]
                node = LayerNode(*nodeInfo)
                node.stream = f"GPU{rank}"

                graph.add_node(node)
                
                for u in last_nodes:
                    u.add_dependency(node)

        for stream, nodes in graph.streams.items():
            if stream == "Comm":
                continue
            for i in range(len(nodes)-1):
                nodes[i].add_dependency(nodes[i+1])

        for nodes in nodes_by_microbatch:
            for i in range(len(nodes)-1):
                nodes[i].add_dependency(nodes[i+1])


    def _add_tp_communication(self, rank, mp, microbatch_idx, feature_map_size):
        comm_node = CommNode(feature_map_size, "Comm")
        comm_node.duration = self.compute_comm_time(comm_node.bucket_size, mp)
        self.graph.add_node(comm_node)
        self.graph.append_node_to_stream(comm_node, f"GPU{rank}")
        self.nodes_by_microbatch[microbatch_idx].append(comm_node)


    def profile(self):
        config = self.config
        
        # collect traces
        log_filename = os.path.join(config.trace_path,
                                    f"trace_{config.hidden_size}_{config.tensor_parallel_size}_{config.micro_batch_size}")
        if os.path.isfile(log_filename):
            with open(log_filename, "r") as f:
                traces = f.readlines()
        else:
            self.create_model()
            trainer = Trainer(config, self.model)
            traces = trainer.train(log_filename)

        # parse traces
        kernel_dict = self.parse_traces(traces)

        return kernel_dict


    def predict(self, kernel_dict):
        graph = self.graph

        # rebuild graph
        for stream, layer_nodes in graph.streams.items():
            # (layer node) ==> (task node)-(task node)-...-(task node)
            new_nodes = []
            for idx, layer_node in enumerate(layer_nodes):
                nodeInfo = kernel_dict.get(layer_node.function, [])
                if len(nodeInfo) == 0:
                    new_nodes.append(layer_node)
                    continue

                # make candidate nodes
                task_nodes = [TaskNode(*(info[:-2] + info[-1:])) for info in nodeInfo]
                for node in task_nodes:
                    node.stream = stream
                for i in range(len(task_nodes)-1):
                    task_nodes[i].add_dependency(task_nodes[i+1])

                # replace nodes
                self.replace_node(layer_node, idx, task_nodes)
                new_nodes += task_nodes

            graph.streams[stream] = new_nodes

        # prediction (Algorithm 1 in paper)
        num_nodes = 0
        Q = []
        P = dict()
        P_brk = dict()
        for stream, nodes in graph.streams.items():
            P[stream] = 0.
            P_brk[stream] = {"compute": 0., "comm": 0.}
            num_nodes += len(nodes)
            for u in nodes:
                if u.ref == 0:
                    Q.append(u)

        logger.info(f"start prediction with {len(Q)} nodes (total {num_nodes} nodes)")

        while Q:
            u = Q.pop(0)
            t = u.stream
            P[t] = max(P[t], u.start + u.duration + u.gap)
            if u.is_comm_node():
                P_brk[t]["comm"] += u.duration
            else:
                P_brk[t]["compute"] += u.duration

            for c in u.child:
                c.start = max(c.start, u.start + u.duration + u.gap)
                c.ref -= 1
                if c.ref == 0:
                    Q.append(c)
                    
        return P, P_brk

    
    def compute_bucket_assignment(self):
        layers = self.layers

        size = 0
        bucket = []
        bucket_indices = []
        bucket_sizes = []
        bucket_size_limit = 1024 * 1024

        for layer_num, (layer_name, _) in reversed(list(enumerate(layers))):
            for p in reversed(self.model_params[layer_name]):
                bucket.append(layer_num)
                size += p.numel() * p.element_size()

                if size >= bucket_size_limit:
                    bucket_indices.append(bucket)
                    bucket_sizes.append(size)
                    size = 0
                    bucket = []
                    bucket_size_limit = self.bucket_size_limit

        if size > 0:
            bucket_indices.append(bucket)
            bucket_sizes.append(size)
        
        return bucket_sizes, bucket_indices


    def parse_traces(self, traces):
        if self.cbid_table is None:
            self.cbid_table = dict()
            self.get_cbid_table()

        cid2func = dict()
        func2node = dict()
        prevFunc = None
        func = "NONE"
        for trace in traces:
            info = trace.strip().split(',')
            type = info[2]
            if type == "TIMESTAMP":
                msg = info[-1].strip('"')
                layer_num = msg.strip().split()[-1]
                if "forward start" in msg:
                    func = f"Fwd_{layer_num}"
                elif "backward start" in msg:
                    func = f"Bwd_{layer_num}"
                elif "WU start" in msg:
                    func = f"WU_{layer_num}"
                elif "end" in msg:
                    func = "NONE"
                
                continue

            if type == "RUNTIME" or type == "DRIVER":
                cid = int(info[-1])
                cid2func[cid] = func

            if type == "KERNEL":
                start = int(info[0])
                duration = int(info[1])
                name = info[3].strip('"')
                cid = int(info[-1])

                if prevFunc and func2node[prevFunc]:
                    prev_task = func2node[prevFunc][-1]
                    func2node[prevFunc][-1] = prev_task[:-1] + (start - prev_task[-2] - prev_task[0],)

                # duration = int(duration / 2.496)
                # duration = int(duration * 0.7)
                nodeInfo = (duration, name, None, cid, start, 0)

                corrFunc = cid2func[cid]
                if corrFunc not in func2node.keys():
                    func2node[corrFunc] = []
                func2node[corrFunc].append(nodeInfo)

                prevFunc = corrFunc
        
        return func2node


    def get_cbid_table(self):
        f = open("src/cupti_runtime_cbid", "r")
        for l in f:
            if not "CUPTI_RUNTIME_TRACE_CBID" in l:
                continue

            api = '_'.join(l.split('_')[4:-1])
            try:
                cbid = int(l.strip().split()[-1][:-1])
            except:
                cbid = int(l.strip().split()[-1][:-1], 16)

            self.cbid_table[cbid] = api

        self.cbid_table[0] = "CUPTI_RUNTIME_TRACE_CBID_INVALID"
        self.cbid_table[336] = "CUPTI_RUNTIME_TRACE_CBID_SIZE"

        f.close()


    def get_allreduce_LUT(self):
        config = self.config

        gpu_name = config.gpu_name.lower()
        base_dir = os.path.join(config.trace_path, gpu_name)
        lut_filenames = [file for file in os.listdir(base_dir) if file.endswith("_LUT")]
        allreduce_LUT = dict()

        for filename in lut_filenames:
            num_gpus = int(filename.split("_")[1][3:])
            allreduce_LUT[num_gpus] = dict()

            f = open(base_dir + "/" + filename, "r")
            lines = f.readlines()[1:]
            f.close()

            for l in lines:
                info = l.strip().split(',')
                size = int(info[0])//1024//1024     # megabytes
                allreduce_LUT[num_gpus][size] = {
                    "time": int(info[-2]),
                    "busbw": float(info[-1])
                }
        
        return allreduce_LUT


    def compute_comm_time(self, size, num_gpus):
        if num_gpus not in self.allreduce_LUT.keys():
            # if there are more than 8 GPUs, latency is estimated by BW
            # assuming a 16-GPU node with all-to-all NVSwitch topology such as HGX
            t = size / (self.config.intra_node_bandwidth * (2 ** 30)) * (2*(num_gpus-1)/num_gpus) # second
            t = t * (10 ** 9)  # nanosecond

        else:
            # read from allreduce latency LUT
            size_mb = round(size / 1024 / 1024)
            if size_mb not in self.allreduce_LUT[num_gpus].keys():
                bw = self.allreduce_LUT[num_gpus][1024]['busbw'] # GB/s
                bw = bw * 1024 # MB/s
                t = size_mb / bw # s
                t = t * (10 ** 9) # ns
            else:
                t = self.allreduce_LUT[num_gpus][size_mb]['time']
        
        return t


    def replace_node(self, old, old_idx, new):
        old_parent = old.parent[:]
        old_child = old.child[:]
        stream = old.stream

        for p in old_parent:
            p.del_dependency(old)
            p.add_dependency(new[0])
        for c in old_child:
            old.del_dependency(c)
            new[-1].add_dependency(c)
        
        for u in new:
            u.function = old.function
        
        new[-1].gap = old.gap
        new[-1].note = old.note


if __name__ == "__main__":

    config = vTrainConfig.load_from_file("test_config.json")
    sim = vTrain()

    result, _ = sim()
    print (result)
    pred_iter_time = max(result.values())/1000/1000
    logging.info(f"predicted iteration time: {pred_iter_time:.6f} ms")
    sim.graph.show_graph()
