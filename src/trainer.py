import os
import torch
import torch.nn as nn

from .model.fused_adam import FusedAdam as Adam

from vtrain_profiler import init_trace, timestamp, finish_trace

import logging

logger = logging.getLogger()


def modify_functions(model):
    def forward_with_info(self, *x):
        timestamp(f"forward start {self.name}")
        ret = self.forward_(*x)
        timestamp(f"forward end {self.name}")
        
        name = self.name

        def backward_pre_hook(self, *args):
            timestamp(f"backward start {name}")

        ret.grad_fn.register_prehook(backward_pre_hook)
        return ret

    def backward_hook(self, *args):
        timestamp(f"backward end {self.name}")

    for _, m in model.named_children():
        m.forward_ = m.forward
        m.forward = forward_with_info.__get__(m, m.__class__)
        if all(p.requires_grad for p in m.parameters()):
            m.register_full_backward_hook(backward_hook)
    
    return model


class Trainer():
    def __init__(self, config, model):
        self.config = config
        self.model = model
        self.layers = [t[-1] for t in model.named_children()]

        for name, layer in self.model.named_children():
            layer.name = f"{name}"

    def train(self, log_filename):
        config = self.config
        local_batch_size = config.micro_batch_size

        model = self.model
        _criterion = nn.CrossEntropyLoss()

        if torch.cuda.is_available():
            model = model.cuda()
            _criterion = _criterion.cuda()

        def criterion(outputs, labels):
            timestamp("forward start loss")
            loss = _criterion(outputs, labels)
            timestamp("forward end loss")
            return loss

        # profile training job
        param_groups = [{"params": l.parameters(), "lr": 0.01, "momentum": 0.9,
                                    "weight_decay": 5e-4, "layer": l.name}
                                for l in self.layers[:-1]]
        optimizer = Adam(param_groups)

        
        logger.info(f"target model: {config.model_arch}")
        logger.info(f"local batch size: {local_batch_size}")

        # warm-up
        model = modify_functions(self.model)
        num_step = 5

        vocab_size = self.model.vocab_size
        inputs = torch.randint(0, vocab_size, (local_batch_size, config.max_length), dtype=int).cuda()
        labels = torch.zeros((local_batch_size, model.vocab_size // config.tensor_parallel_size), dtype=int).cuda()

        # warmup
        torch.backends.cudnn.benchmark = True
        for _ in range(10):
            self.train_step(model, inputs, labels, criterion, optimizer)
            torch.cuda.empty_cache()
        torch.backends.cudnn.benchmark = False

        for _ in range(num_step):
            self.train_step(model, inputs, labels, criterion, optimizer)
        torch.cuda.synchronize()

        # collect traces
        init_trace()

        self.train_step(model, inputs, labels, criterion, optimizer, profile=True)

        torch.cuda.synchronize()
        traces = finish_trace().strip().split("\n")
        traces.sort(key=lambda l: int(l.split(',')[0]))

        logger.info(f"number of traces collected: {len(traces)}")

        with open(log_filename, "w") as f:
            f.write("\n".join(traces))

        return traces

        

    def train_step(self, model, inputs, labels, criterion, optimizer, profile=False):
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        loss.backward()

        optimizer.step(profile=profile)
        optimizer.zero_grad()