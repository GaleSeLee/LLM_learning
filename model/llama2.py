from torch import nn
import torch

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
        h = self.tok_embeddings(tokens)
        self.freqs_cis = self.freqs_cis.to(h.device)
        freqs_cis = self.freqs_cis[start_pos : start_ops+seq_len]


