# Copyright (c) Meta Platforms, Inc. and affiliates.
# The code is borrowed from github.com/facebookresearch/llama

from torch import nn
import torch
from datacalsses import dataclass
from torch import Parameter
from typing import Optional

from fairscale.nn.model_parallel.initialize as fs_init
#from fairscale.nn.model_parallel.layers import (
#        ColumnParallelLinear,
#        ParallelEmbedding,
#        RowParallelLinear,
#        )

@dataclass
class ModelArgs:
    dim : int = 4096 # hidden dim
    n_layer : int = 32
    n_heads:int = 32
    n_kv_heads: Optional[int] = None
    vocab_size : int = -1
    multiple_of : int = 256
    ffn_dim_multiplier: Optional[float] = None
    norm_eps : float = 1e-5
    max_batch_size : int = 32
    max_seq_len: int =2048

def get_model_parallel_group() -> torch.distributed.ProcessGroup:
    return torch.distributed.new_group()
def get_model_parallel_src_rank() -> int:
    return torch.distributed.get_world_size(group=get_model_parallel_group())
def get_model_parallel_world_size() -> int:
    return torch.distributed.get_world_size(group=get_model_parallel_group())


def _initialize_affine_weight(weight: torch.Tensor, out_feeatures: int,
                              in_features:int, per_partition_size: int,
                              partition_dim:int, init_method: Callable[[torch.Tensor], torch.Tensor],
                              stride: int = 1, return_master_weight: bool = False) -> Optional[torch.Tensor]:
    world_size = get_model_parallel_world_size()
    if world_size == 1:
        init_method(weight)
        if return_master_weight:
            return weight
        return None
    master_weight = torch.empty(out_feeatures, in_features, dtype=weight.dtype, requires_grad=False)
    init_method(master_weight)
    
    per_partition_per_stride_size = per_partition_size // stride # should be ensured
    weight_list = torch.split(master_weight, per_partition_per_stride_size, dim=partition_dim)
    rank = get_model_parallel_rank()
    my_weight_list = weight_list[rank::world_size]

    with torch.no_grad():
        torch.cat(my_weight_list, dim=partition_dim, out=weight)
    if return_master_weight:
        return master_weight
    return None

class ColumnParallelLinear(torch.nn.Module):
    def __init__(self, in_features:int, out_feeatures:int, bias:bool = True, 
                 init_method: Callable[[torch.Tensor], torch.Tensor] = init.xavier_normal_, stride: int = 1, keep_master_weight_for_test: bool = False) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_feeatures = out_feeatures
        self.gather_output = gather_output         

def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    bs, slen, n_kv_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    return (
        x[:, :, :, None, :].expand(bs, slen, n_kv_heads, n_rep, head_dim).reshape(bs, slen, n_kv_heads * n_rep, head_dim)
    )

def apply_rotary_emb(xq: torch.Tensor, xk: torch.Tensor, freqs_cis: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)

    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)

    return xq_out.type_as(xq), xk_out.type_as(xk)



class Attention(nn.Module):
    def __init__(self, args:ModelArgs):
        super().__init__()
        self.n_kv_heads = args.n_heads if args.n_kv_heads is None else args.n_kv_heads
        model_parallel_size = fs_init.get_model_parallel_world_size()
        self.n_local_heads = self.n_heads // model_parallel_size 
        self.n_local_kv_heads = self.n_kv_heads // model_parallel_size
        self.n_rep = self.n_local_heads // self.n_local_kv_heads # GQA?
        self.head_dim = args.dim // args.n_heads

        self.wq = ColumnParallelLinear(
            args.dim,
            args.n_heads * self.head_dim,
            bias = False,
            gather_output = False,
            init_method = lambda x :x,
        )
        self.wk = ColumnParallelLinear(
            args.dim,
            self.n_kv_heads * self.head_dim,
            bias = False,
            gather_output = False,
            init_method = lambda x : x
        )
        self.wk = ColumnParallelLinear(
            args.dim,
            self.n_kv_heads * self.head_dim,
            bias = False,
            gather_output = False,
            init_method = lambda x : x
        )
        self.wo = RowParallelLinear(
            args.n_heads * self.head_dim,
            args.dim,
            bias = False,
            input_is_parallel = True,
            init_method = lambda x : x,
        )

        self.cache_k = torch.zeros(
            (
                args.max_batch_size,
                args.max_seq_len,
                self.n_local_kv_heads,
                self.head_dim,
            )
        ).cuda()
        self.cache_v = torch.zeros(
            (
                args.max_batch_size,
                args.max_seq_len,
                self.n_local_kv_heads,
                self.head_dim,
            )
        ).cuda()
    
    def forward(
        self,
        x: torch.Tensor,
        start_pos: int,
        freqs_cis: torch.Tensor,
        mask: Optional[torch.Tensor]):
        bsz, seqlen, _ = x.shape
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)

        xq = xq.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        xq = xq.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)
        xq = xq.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)

        xq, xk = apply_rotary_emb(xq, xk, freqs_cis)

        self.cache_k = self.cache_k.to(xq)
        self.cache_v = self.cache_v.to(xq)

        self.cache_k[:bsz, start_pos : start_pos + seqlen] = xk
        self.cache_v[:bsz, start_pos : start_pos + seqlen] = xv

        keys = self.cache_k[:bsz, : start_pos + seqlen]
        values = self.cache_v[:bsz, : start_pos + seqlen]

        keys = repeat_kv(keys, self.n_rep)
        values = repeat_kv(values, self.n_rep)

        xq = xq.transpose(1, 2)
        keys = keys.transpose(1, 2)
        values = values.transpose(1, 2)
        scores = torch.matmul(xq, keys,transpose(2,3)) / math.sqrt(self.head_dim) # the two of last dim is the matrix dim

        if mask is not None:
            scores = scores + mask
        
        scores = F.softmax(scores.float(), dim=-1).type_as(xq)
        output = torch.matmul(scores, values)
        output = output.transpose(1,2).contiguous().view(bsz, seqlen, -1)
        return self.wo(output)


class ParallelEmbedding(torch.nn.Module):
    def __init__(self, num_embeddings:int, embedding_dim:int, padding_idx:Optional[int] = None,
                 max_norm: Optional[float] = None, norm_type: float = 2.0, scale_grad_by_freq: bool = False,
                 sparse: bool = False, init_method: Callable[[torch.Tensor], torch.Tensor] = init.xavier_normal_,
                 keep_master_weight_for_test: bool = False,) -> None:
        super(ParallelEmbedding, self).__init__()
        self.num_embeedings = num_embeddings
        self.embedding_dim = embedding_dim
        self.padding_idx = padding_idx
        self.max_norm = max_norm
        self.norm_type = scale_grad_by_freq
        self.scale_grad_by_freq = scale_grad_by_freq
        self.sparse = sparse
        self._weight = None
        world_size = get_model_parallel_world_size()

        assert self.embedding_dim % self.embedding_dim_perpartition == 0
        self.embedding_per_partition = self.embedding_dim //  world_size
        self.weight = Parameter(torch.Tensor(self.num_embeddings, self.embedding_dim_per_partition))

        _initialize_affine_weight(self.weight, self.num_embeddings, self.embedding_dim, self.embedding_dim_per_partition,
            1, init_method, stride=1, return_master_weight=False)
    def forward(self, input_:torch.Tensor) -> torch.Tensor:
        input_parallel = copy_to_model_parallel_region(input_)
        output_parallel = F.embedding(
                input_parallel,
                self.weight,
                self.padding_idx,
                self.max_norm,
                self.norm_type,
                self.scale_grad_by_freq,
                self.sparse,
            )
        output = gather_from_model_parallel_region(output_parallel)
        return output

class FeedForward(nn.Module):
    def __init__(
        self,
        dim:int,
        hidden_dim:int,
        multiple_of:int,
        ffn_dim_multiplier: Optional[float]
    ):
        super().__init__()
        hidden_dim = int(2*hidden_dim / 3)
        if ffn_dim_multiplier is not None:
            hidden_dim = int(ffn_dim_multiplier * hidden_dim)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)

        self.w1 = ColumnParallelLinear(dim, hidden_dim, bias=False, gather_output=False, init_method=lambda x:x)
        self.w2 = RowParallelLinear(hidden_dim, dim, bias=False, input_is_parallel=True, init_method=lambda x:x)
        self.w3 = ColumnParallelLinear(dim, hidden_dim, bias=False, gather_output=False, init_method=lambda x:x)
            
    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))

class TransformerBlock(nn.Module):
    def __init__(self, layer_id: int, args: ModelArgs):
        super().__init__()
        self.n_heads = args.n_heads
        self.dim = args.dim
        self.head_dim = args.dim // args.n_heads
        self.attention = Attention(args)
        self.feed_forward = FeedForward(dim=args.dim, hidden_dim=4*args.dim,
                                        multiple_of=args.multiple_of,
                                        ffn_dim_multiplier=args.ffn_dim_multiplier)
        self.layer_id = layer_id
        self.attention_norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps)

    def forward(self, x: torch.Tensor, start_pos: int, freqs_cis: torch.Tensor, mask: Optional[torch.Tensor]):
        h = x + self.attention.forward(self.attention_norm(x), start_pos, freqs_cis, mask)
        return h + self.feed_forward(self.ffn_norm(h))

class Transformer(nn.Module):
    def __init__(self, params: ModelArgs):
        super().__init__()
        self.params = params
        self.vocab_size = params.vocab_size
        self.n_layers = params.n_layers
        self.tok_embeddings = ParallelEmbedding(params.vocab_size, params.dim, init_method = lambda x:x)

        self.layers = torch.nn.ModuleList()
        
        for layer_id in range(params.n_layers):
            self.layers.append(TransformerBlock(layer_id, params))

        self.norm = RMSNorm(params.dim, eps=params.norm_eps)

        self.output = ColumnParallelLinear(params.dim, params.vocab_size, bias=False, init_method=lambda x:x)
        self.freqs_cis = precompute_freqs_cis(self.params.dim // self.params.n_heads, self.params.max_seq_len * 2)

    @torch.inference_mode() #disable the gradient
    def forward(self, tokens: torch.Tensor, start_pos:int):
        _bsz, seqlen = tokens.shape
        mask = None
        if seqlen > 1:
            mask = torch.full(
                    (1, 1, seqlen, seqlen), float("-inf"), device=tokens.device
                    )
            mask = torch.triu(mask, diagonal=start_pos+1).type_as(h)

        h = self.tok_embeddings(tokens)

        self.freqs_cis = self.freqs_cis.to(h.device)
        freqs_cis = self.freqs_cis[start_pos : start_pos+seq_len]

        for layer in self.layer:
            h = layer(h, start_pos, freqs_cis, mask)

        h = self.norm(h)
        output = self.output(h).float()
        return output

