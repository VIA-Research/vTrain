import torch
import torch.nn as nn
import torch.nn.functional as F

import math

from .utils import divide
from .utils import split_tensor_along_last_dim

from .layers import ColumnParallelLinear
from .layers import RowParallelLinear

from apex.normalization import MixedFusedLayerNorm as LayerNorm
from .fused_softmax import FusedScaleMaskSoftmax
from .fused_bias_gelu import bias_gelu_impl


def bias_dropout_add(x, bias, residual, prob, training):
    # type: (Tensor, Tensor, Tensor, float, bool) -> Tensor
    out = torch.nn.functional.dropout(x + bias, p=prob, training=training)
    out = residual + out
    return out


def get_bias_dropout_add(training):
    def _bias_dropout_add(x, bias, residual, prob):
        return bias_dropout_add(x, bias, residual, prob, training)
    return _bias_dropout_add


@torch.jit.script
def bias_dropout_add_fused_train(x, bias, residual, prob):
    # type: (Tensor, Tensor, Tensor, float) -> Tensor
    return bias_dropout_add(x, bias, residual, prob, True)


class ShardedGptLogit(torch.nn.Module):
    def __init__(self, vocab_size, hidden_size, world_size):
        super(ShardedGptLogit, self).__init__()
        # self.embedding_weight = embedding_weight
        self.embedding_weight = nn.Embedding(vocab_size // world_size, hidden_size).weight
        self.world_size = world_size
    
    def forward(self, transformer_output):
        # print (transformer_output.size(), self.embedding_weight.size())
        # print (transformer_output.shape, self.embedding_weight.shape)
        # exit()
        logits = F.linear(transformer_output, self.embedding_weight)
        # logits = torch.cat([logits for _ in range(self.world_size)], -1) 

        return logits


class ShardedGptEmbeddings(torch.nn.Module):
    def __init__(self, vocab_size, hidden_size, world_size,
                 max_sequence_length, embedding_dropout_prob):
        super(ShardedGptEmbeddings, self).__init__()
        self.vocab_size = vocab_size
        self.world_size = world_size
        self.max_sequence_length = max_sequence_length

        self.word_embeddings = nn.Embedding(vocab_size // world_size, hidden_size)
        self.position_embeddings = nn.Embedding(max_sequence_length, hidden_size)

        # embeddings dropout
        self.embedding_dropout = torch.nn.Dropout(embedding_dropout_prob)
    
    def forward(self, input_ids, position_ids=None):
        seq_length = self.max_sequence_length
        local_vocab_size = self.vocab_size // self.world_size

        if position_ids is None:
            position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)
            position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
        
        input_ids_mask = input_ids >= local_vocab_size
        masked_input_ids = input_ids.clone()
        masked_input_ids[input_ids_mask] = 0
        words_embeddings = self.word_embeddings(masked_input_ids)        
        position_embeddings = self.position_embeddings(position_ids)
        embeddings = words_embeddings + position_embeddings

        # Dropout.
        embeddings = self.embedding_dropout(embeddings)

        return embeddings


class ShardedGptMlp(torch.nn.Module):
    def __init__(self, hidden_size, world_size, output_dropout_prob):
        super(ShardedGptMlp, self).__init__()

        # Project to 4h.
        self.dense_h_to_4h = ColumnParallelLinear(hidden_size, 4*hidden_size,
                                                  world_size, gather_output=False,
                                                  skip_bias_add=True)
        # Project back to h.
        self.dense_4h_to_h = RowParallelLinear(
            4*hidden_size,
            hidden_size,
            world_size,
            input_is_parallel=True,
            skip_bias_add=True)

    def forward(self, hidden_states):
        # [b, s, 4hp]
        intermediate_parallel, bias_parallel = self.dense_h_to_4h(hidden_states)
        # intermediate_parallel = F.gelu(intermediate_parallel + bias_parallel)
        intermediate_parallel = bias_gelu_impl(intermediate_parallel, bias_parallel)

        # [b, s, h]
        output, output_bias = self.dense_4h_to_h(intermediate_parallel)
        return output, output_bias


class ShardedGptSelfAttention(torch.nn.Module):
    def __init__(self, hidden_size,
                 world_size,
                 num_attention_heads,
                 attention_dropout_prob,
                 output_dropout_prob,
                 layer_number):
        super(ShardedGptSelfAttention, self).__init__()

        self.layer_number = layer_number

        # Per attention head and per partition values.
        self.hidden_size_per_partition = divide(hidden_size, world_size)
        self.hidden_size_per_attention_head = divide(hidden_size,
                                                     num_attention_heads)
        self.num_attention_heads_per_partition = divide(num_attention_heads,
                                                        world_size)
        # Strided linear layer.
        self.query_key_value = ColumnParallelLinear(hidden_size, 3*hidden_size,
                                                    world_size, gather_output=False)
        
        masked_softmax_fusion = True
        def attention_mask_func(attention_scores, attention_mask):
            attention_scores.masked_fill_(attention_mask.bool(), -10000.0)
            return attention_scores
        
        coeff = None
        self.norm_factor = math.sqrt(self.hidden_size_per_attention_head)
        coeff = self.layer_number
        self.norm_factor *= coeff

        self.scale_mask_softmax = FusedScaleMaskSoftmax(
            True, False,
            masked_softmax_fusion,
            attention_mask_func,
            True,
            coeff)

        # Dropout. Note that for a single iteration, this layer will generate
        # different outputs on different number of parallel partitions but
        # on average it should not be partition dependent.
        self.attention_dropout = torch.nn.Dropout(attention_dropout_prob)

        # Output.
        self.dense = RowParallelLinear(hidden_size, hidden_size,
                                       world_size, input_is_parallel=True,
                                       skip_bias_add=True)
        self.output_dropout = torch.nn.Dropout(output_dropout_prob)
    
    def _transpose_for_scores(self, tensor):
        """Transpose a 3D tensor [b, s, np*hn] into a 4D tensor with
        size [b, np, s, hn].
        """
        new_tensor_shape = tensor.size()[:-1] + \
                           (self.num_attention_heads_per_partition,
                            self.hidden_size_per_attention_head)
        tensor = tensor.view(*new_tensor_shape)
        return tensor.permute(0, 2, 1, 3)

    def forward(self, hidden_states, attention_mask):
        # hidden_states: [sq, b, h]

        # =====================
        # Query, Key, and Value
        # =====================

        # Attention heads [sq, b, h] --> [sq, b, (np * 3 * hn)]
        mixed_x_layer, _ = self.query_key_value(hidden_states)

        # [sq, b, (np * 3 * hn)] --> [sq, b, np, 3 * hn]
        new_tensor_shape = mixed_x_layer.size()[:-1] + \
            (self.num_attention_heads_per_partition,
                3 * self.hidden_size_per_attention_head)
        mixed_x_layer = mixed_x_layer.view(*new_tensor_shape)

        # [sq, b, np, 3 * hn] --> 3 [sq, b, np, hn]
        (query_layer,
            key_layer,
            value_layer) = split_tensor_along_last_dim(mixed_x_layer, 3)
        
        # ===================================
        # Raw attention scores. [b, np, s, s]
        # ===================================

        # [b, np, sq, sk]
        output_size = (query_layer.size(1),
                       query_layer.size(2),
                       query_layer.size(0),
                       key_layer.size(0))

        # [sq, b, np, hn] -> [sq, b * np, hn]
        query_layer = query_layer.view(output_size[2],
                                       output_size[0] * output_size[1], -1)
        # [sk, b, np, hn] -> [sk, b * np, hn]
        key_layer = key_layer.view(output_size[3],
                                   output_size[0] * output_size[1], -1)

        # preallocting result tensor: [b * np, sq, sk]
        matmul_result = torch.empty(
            output_size[0]*output_size[1],
            output_size[2],
            output_size[3],
            dtype=query_layer.dtype,
            device=torch.cuda.current_device())

        # Raw attention scores. [b * np, sq, sk]
        matmul_result = torch.baddbmm(
            matmul_result,
            query_layer.transpose(0, 1),   # [b * np, sq, hn]
            key_layer.transpose(0, 1).transpose(1, 2),  # [b * np, hn, sk]
            beta=0.0, alpha=(1.0/self.norm_factor))

        # change view to [b, np, sq, sk]
        attention_scores = matmul_result.view(*output_size)

        # ===========================
        # Attention probs and dropout
        # ===========================

        # attention scores and attention mask [b, np, sq, sk]
        attention_probs = self.scale_mask_softmax(attention_scores,
                                                  attention_mask)
        attention_probs = self.attention_dropout(attention_probs)


        # =========================
        # Context layer. [sq, b, hp]
        # =========================

        # value_layer -> context layer.
        # [sk, b, np, hn] --> [b, np, sq, hn]

        # context layer shape: [b, np, sq, hn]
        output_size = (value_layer.size(1),
                       value_layer.size(2),
                       query_layer.size(0),
                       value_layer.size(3))

        # change view [sk, b * np, hn]
        value_layer = value_layer.view(value_layer.size(0),
                                       output_size[0] * output_size[1], -1)

        # change view [b * np, sq, sk]
        attention_probs = attention_probs.view(output_size[0] * output_size[1],
                                               output_size[2], -1)

        # matmul: [b * np, sq, hn]
        context_layer = torch.bmm(attention_probs, value_layer.transpose(0, 1))

        # change view [b, np, sq, hn]
        context_layer = context_layer.view(*output_size)

        # [b, np, sq, hn] --> [sq, b, np, hn]
        context_layer = context_layer.permute(2, 0, 1, 3).contiguous()

        # [sq, b, np, hn] --> [sq, b, hp]
        new_context_layer_shape = context_layer.size()[:-2] + \
            (self.hidden_size_per_partition,)
        context_layer = context_layer.view(*new_context_layer_shape)

        # =================
        # Output. [sq, b, h]
        # =================

        output, bias = self.dense(context_layer)

        return output, bias


class ShardedGptTransformerLayer(torch.nn.Module):
    def __init__(self, hidden_size,
                 world_size,
                 num_attention_heads,
                 attention_dropout_prob,
                 output_dropout_prob,
                 layernorm_epsilon,
                 layer_number):
        super(ShardedGptTransformerLayer, self).__init__()

        self.layer_number = layer_number

        self.input_layernorm = LayerNorm(hidden_size, eps=layernorm_epsilon)

        self.attention = ShardedGptSelfAttention(
                            hidden_size,
                            world_size,
                            num_attention_heads,
                            attention_dropout_prob,
                            output_dropout_prob,
                            layer_number=layer_number)
        
        self.post_attention_layernorm = LayerNorm(hidden_size, eps=layernorm_epsilon)

        self.mlp = ShardedGptMlp(
                        hidden_size,
                        world_size,
                        output_dropout_prob)
    
    def forward(self, hidden_states, ltor_mask):
        # hidden_states: [b, s, h]
        # ltor_mask: [1, 1, s, s]

        # Layer norm at the begining of the transformer layer.
        layernorm_output = self.input_layernorm(hidden_states)
        # Self attention.
        attention_output, attention_bias = self.attention(layernorm_output, ltor_mask)
        bias_dropout_add_func = bias_dropout_add_fused_train
        # bias_dropout_add_func = get_bias_dropout_add(True)
        layernorm_input = bias_dropout_add_func(
                attention_output,
                attention_bias.expand_as(hidden_states),
                hidden_states,
                0.1)
        # Layer norm post the self attention.
        layernorm_output = self.post_attention_layernorm(layernorm_input)
        # MLP.
        mlp_output, mlp_bias = self.mlp(layernorm_output)
        # Second residual connection.
        output = bias_dropout_add_func(
                    mlp_output,
                    mlp_bias.expand_as(layernorm_input),
                    layernorm_input,
                    0.1)

        return output
        

class ShardedGptTransformer(torch.nn.Module):
    def __init__(self, num_layers,
                 hidden_size,
                 world_size,
                 num_attention_heads,
                 attention_dropout_prob,
                 output_dropout_prob,
                 checkpoint_activations,
                 checkpoint_num_layers=1,
                 layernorm_epsilon=1e-5):
        super(ShardedGptTransformer, self).__init__()

        self.checkpoint_activations = checkpoint_activations
        self.checkpoint_num_layers = checkpoint_num_layers

        self.layers = torch.nn.ModuleList(
                            [
                                ShardedGptTransformerLayer(
                                    hidden_size,
                                    world_size,
                                    num_attention_heads,
                                    attention_dropout_prob,
                                    output_dropout_prob,
                                    layernorm_epsilon,
                                    layer_number=layer_number+1
                                )
                                for layer_number in range(num_layers)
                            ]
                        )


    def forward(self, hidden_states, attention_mask):
        hidden_states = hidden_states.transpose(0, 1).contiguous()
        for layer in self.layers:
            hidden_states = layer(hidden_states, attention_mask)

        output = hidden_states.transpose(0, 1).contiguous()

        return output
