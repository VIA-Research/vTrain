from typing import Optional, Any
import json


class vTrainConfig:
    """
    Configuration class for the framework.
    
    Attributes:
        num_gpus (int): Number of total GPUs.
        tensor_parallel_size (int): Tensor parallel size.
        data_parallel_size (int): Data parallel size.
        pipeline_parallel_size (int): Pipeline parallel size.
        gpu_name (str): Name of the GPU to use
        global_batch_size (int): Global batch size.
        micro_batch_size (int): Micro-batch size.
        mnodel_arch (str): Name of PyTorch class of the target model architecture
        num_layers (int): Number of transformer layers.
        hidden_size (int): The size of hidden dimension.
        num_attention_heads (int): Number of attention heads.
        max_length (int): Maximum sequence length.
        use_gradient_bucket (bool): Whether PyTorch DDP's gradient bucketing is used or not.
        use_checkpoint (bool): Whether activation checkpointing is used or not.
        ddp_bucket_size (int): The size of gradient bucket used in PyTorch DDP (bytes).
        inter_node_bandwidth (int): Total bandwidth of inter-node communication in Gbps.
        intra_node_bandwidth (int): Total bandwidth of intra-node communication in GB/s.
        pipeline_scheduling (str): Pipeline scheduling (default: "1f1b")
        node_size (int): Number of GPUs within a node.
        trace_path (str): Path where GPU kernel traces exist and are going to be stored.
    """
    
    def __init__(self,
                 num_gpus: Optional[int]                    = None,
                 tensor_parallel_size: Optional[int]        = None,
                 data_parallel_size: Optional[int]          = None,
                 pipeline_parallel_size: Optional[int]      = None,
                 gpu_name: str                              = "A100",
                 global_batch_size: int                     = 1920,
                 micro_batch_size: Optional[int]            = None,
                 model_arch: str                            = "ShardedGptModel",
                 num_layers: int                            = 105,
                 hidden_size: int                           = 20480,
                 num_attention_heads: int                   = 128,
                 max_length: int                            = 2048,
                 vocab_size: int                            = 50257,
                 use_gradient_bucket: bool                  = False,
                 use_checkpoint: bool                       = True,
                 ddp_bucket_size: Optional[int]             = None,                 # MB
                 inter_node_bandwidth: int                  = 800,                  # Gbps
                 intra_node_bandwidth: int                  = 150,                  # GB/s
                 pipeline_scheduling: str                   = "1f1b",
                 node_size: int                             = 8,
                 trace_path: str                            = "trace/"
                 ):
        
        self.num_gpus = num_gpus
        self.gpu_name = gpu_name
        self.tensor_parallel_size = tensor_parallel_size
        self.data_parallel_size = data_parallel_size
        self.pipeline_parallel_size = pipeline_parallel_size
        self.global_batch_size = global_batch_size
        self.micro_batch_size = micro_batch_size
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.max_length = max_length
        self.vocab_size = vocab_size
        self.use_gradient_bucket = use_gradient_bucket
        self.use_checkpoint = use_checkpoint
        self.ddp_bucket_size = ddp_bucket_size
        self.pipeline_scheduling = pipeline_scheduling
        self.inter_node_bandwidth = inter_node_bandwidth
        self.intra_node_bandwidth = intra_node_bandwidth
        self.node_size = node_size
        self.trace_path = trace_path
        
        # target model
        self.model_arch = model_arch
        
        # validate configuration
        self.validate_config()
            

    def validate_config(self):
        """Validate configuration constraints."""
        
        if self.num_gpus is None:
            assert all(x is not None for x in [self.tensor_parallel_size, self.data_parallel_size, self.pipeline_parallel_size])
            self.num_gpus = self.tensor_parallel_size * self.pipeline_parallel_size * self.data_parallel_size
        elif any(x is None for x in [self.tensor_parallel_size, self.data_parallel_size, self.pipeline_parallel_size]):
            if sum(x is None for x in [self.tensor_parallel_size, self.data_parallel_size, self.pipeline_parallel_size]) > 1:
                raise AssertionError("Only one of tensor_parallel_size, data_parallel_size, or pipeline_parallel_size can be None.")
            
            # Calculate the missing value
            if self.tensor_parallel_size is None:
                self.tensor_parallel_size = self.num_gpus // (self.pipeline_parallel_size * self.data_parallel_size)
            elif self.data_parallel_size is None:
                self.data_parallel_size = self.num_gpus // (self.tensor_parallel_size * self.pipeline_parallel_size)
            elif self.pipeline_parallel_size is None:
                self.pipeline_parallel_size = self.num_gpus // (self.tensor_parallel_size * self.data_parallel_size)
        else:
            assert self.num_gpus == (self.tensor_parallel_size * self.pipeline_parallel_size * self.data_parallel_size), \
                "num_gpus must be equivalent to (tensor_parallel_size * pipeline_parallel_size * data_parallel_size)."
        
        if self.pipeline_parallel_size <= 1:
            self.micro_batch_size = self.global_batch_size // self.data_parallel_size
            print (f"[vTrainConfig] micro_batch_size is set by 'global_batch_size // data_parallel_size' as pipeline_parallel_size is 1")

            
        assert self.global_batch_size % self.data_parallel_size == 0, \
            "global_batch_size must be divisible by data_parallel_size."
        assert self.global_batch_size % self.micro_batch_size == 0, \
            "global_batch_size must be divisible by micro_batch_size."
        assert self.num_layers % self.pipeline_parallel_size == 0, \
            "num_layers must be divisible by pipeline_parallel_size."
        assert self.hidden_size % self.num_attention_heads == 0, \
            "hidden_size must be divisible by num_attention_heads."
        assert self.num_attention_heads % self.tensor_parallel_size == 0, \
            "num_attention_heads must be divisible by tensor_parallel_size."
        

    def save_to_file(self, file_path: str):
        """Save configuration to a JSON file."""
        with open(file_path, 'w') as f:
            json.dump(self.__dict__, f, indent=4)


    @classmethod
    def load_from_file(cls, file_path: str):
        """Load configuration from a JSON file."""
        with open(file_path, 'r') as f:
            config_dict = json.load(f)
        return cls(**config_dict)
    

    def __repr__(self):
        return (
            "vTrainConfig(\n"
            f"  num_gpus={self.num_gpus},\n"
            f"  gpu_name='{self.gpu_name}',\n"
            f"  tensor_parallel_size={self.tensor_parallel_size},\n"
            f"  data_parallel_size={self.data_parallel_size},\n"
            f"  pipeline_parallel_size={self.pipeline_parallel_size},\n"
            f"  global_batch_size={self.global_batch_size},\n"
            f"  micro_batch_size={self.micro_batch_size},\n"
            f"  model_arch='{self.model_arch}',\n"
            f"  num_layers={self.num_layers},\n"
            f"  hidden_size={self.hidden_size},\n"
            f"  num_attention_heads={self.num_attention_heads},\n"
            f"  max_length={self.max_length},\n"
            f"  use_gradient_bucket={self.use_gradient_bucket},\n"
            f"  use_checkpoint={self.use_checkpoint},\n"
            f"  ddp_bucket_size={self.ddp_bucket_size},\n"
            f"  inter_node_bandwidth={self.inter_node_bandwidth},\n"
            f"  intra_node_bandwidth={self.intra_node_bandwidth},\n"
            f"  pipeline_scheduling='{self.pipeline_scheduling}',\n"
            f"  node_size={self.node_size},\n"
            f"  trace_path='{self.trace_path}'\n"
            ")"
        )

    

if __name__ == "__main__":
    import os

    config_dir = "../config/"
    test_configs = os.listdir(config_dir)

    for test_config in test_configs:
        if not test_config.startswith("config_test"):
            continue

        test_name = "_".join(test_config.split(".")[0].split("_")[2:])
        print ("="*12 + f" TEST: {test_name} " + "="*12)

        try:
            config = vTrainConfig.load_from_file(config_dir + test_config)
        except AssertionError as e:
            print (f"AssertionError occurred: {e}")
        else:
            print (config)

        print ()