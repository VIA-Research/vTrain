import torch
import torch.nn as nn

from .gpt_modeling import ShardedGptTransformer
from .gpt_modeling import ShardedGptEmbeddings
from .gpt_modeling import ShardedGptLogit

from apex.normalization import FusedLayerNorm as LayerNorm

class ShardedGptModel(nn.Module):
    def __init__(self, num_layers,
                 hidden_size,
                 world_size,
                 vocab_size=50257,
                 num_attention_heads=16,
                 embedding_dropout_prob=0.1,
                 attention_dropout_prob=0.1,
                 output_dropout_prob=0.1,
                 max_sequence_length=1024,
                 checkpoint_activations=False,
                 checkpoint_num_layers=1):
        
        super(ShardedGptModel, self).__init__()
        self.vocab_size = vocab_size
        self.world_size = world_size
        self.max_sequence_length = max_sequence_length

        # embeddings
        self.embeddings = ShardedGptEmbeddings(vocab_size, hidden_size, world_size,
                                               max_sequence_length, embedding_dropout_prob)
        
        # transformer
        self.transformer = ShardedGptTransformer(num_layers,
                                                 hidden_size,
                                                 world_size,
                                                 num_attention_heads,
                                                 attention_dropout_prob,
                                                 output_dropout_prob,
                                                 checkpoint_activations,
                                                 checkpoint_num_layers)
        
        self.layernorm = LayerNorm(hidden_size, eps=1e-5)
        
        # logits
        self.logit = ShardedGptLogit(self.vocab_size, hidden_size, world_size)

    
    def forward(self, input_ids, position_ids=None, attention_mask=None):
        seq_length = self.max_sequence_length

        if attention_mask is None:
            attention_mask = torch.tril(torch.ones(
                            (input_ids.shape[0], seq_length, seq_length), device=input_ids.device)).view(
                                input_ids.shape[0], 1, seq_length, seq_length).half()

        # Embeddings.
        embeddings = self.embeddings(input_ids, position_ids)

        # Transformer.
        transformer_output = self.transformer(embeddings, attention_mask)
        transformer_output = self.layernorm(transformer_output)

        # Logits
        output = self.logit(transformer_output)

        return output
        

if __name__ == "__main__":
    # import torch.cuda.nvtx as nvtx

    hidden_size = 1024
    vocab_size = 50257
    num_hidden_layers = 1
    world_size = 2

    batch_size = 2
    max_position_length = 1024

    # bert = ShardedBertModel(config)
    gpt = ShardedGptModel(num_layers=1, hidden_size=hidden_size, world_size=4,
                            max_sequence_length=max_position_length)

    input_ids = torch.randint(0, vocab_size, (batch_size, max_position_length))

    input_ids = input_ids.cuda()
    gpt = gpt.cuda()

    for i in range(30):
        output = gpt(input_ids)
        torch.cuda.synchronize()
        exit()

    # print (f"bert output shape: {output[0].shape}")
