import torch


class LayerNorm(torch.nn.Module):
    def __init__(self, ndim: int, bias: bool):
        super(LayerNorm, self).__init__()
        self.weight = torch.nn.Parameter(torch.ones(ndim))
        self.bias = torch.nn.Parameter(torch.zeros(ndim)) if bias else None
    
    def forward(self, x: torch.Tensor):
        return torch.nn.functional.layer_norm(x, self.weight.shape, self.weight, self.bias, 1e-5)
    

class MLP(torch.nn.Module):
    def __init__(self, edim: int, dropout: float, bias: bool):
        super(MLP, self).__init__()
        self.expansion = torch.nn.Linear(edim, 4*edim, bias)
        self.activation = torch.nn.GELU()
        self.projection = torch.nn.Linear(edim*4, edim, bias)
        self.dropout = torch.nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor): 
        return self.dropout(self.projection(self.activation(self.expansion(x))))
    

class SelfAttention(torch.nn.Module):
    def __init__(self, edim: int, n_head: int, dropout: bool, bias: bool, causal: bool, blocksize: int):
        super(SelfAttention, self).__init__()
        self.qkv = torch.nn.Linear(edim, 3*edim, bias)
        self.projection = torch.nn.Linear(edim, edim, bias)

        self.attn_dropout = torch.nn.Dropout(dropout)
        self.resid_dropout = torch.nn.Dropout(dropout)
        self.n_head = n_head
        self.edim = edim
        self.dropout = dropout
        self.causal = causal
        self.blocksize = blocksize
    
    def forward(self, x: torch.Tensor, mask: torch.Tensor = None):
        Bs,Tn,Ed = x.shape
        q,k,v = self.qkv(x).split(self.edim, 2)

        q = q.view(Bs, Tn, self.n_head, Ed // self.n_head).transpose(1,2)
        k = k.view(Bs, Tn, self.n_head, Ed // self.n_head).transpose(1,2)
        v = v.view(Bs, Tn, self.n_head, Ed // self.n_head).transpose(1,2)

        y = torch.nn.functional.scaled_dot_product_attention(q,k,v,mask,
            dropout_p=self.dropout if self.training else 0,
            is_causal=self.causal)
        
        y = y.transpose(1,2).contiguous().view(Bs, Tn, Ed)
        return self.resid_dropout(self.projection(y))
    

class Block(torch.nn.Module):
    def __init__(self, edim: int, n_head: int, dropout: float, bias: bool, causal: bool, blocksize: int):
        super(Block, self).__init__()
        self.ln1 = LayerNorm(edim, bias)
        self.attention = SelfAttention(edim, n_head, dropout, bias, causal, blocksize)
        self.ln2 = LayerNorm(edim, bias)
        self.mlp = MLP(edim, dropout, bias)
    
    def forward(self, x: torch.Tensor, mask: torch.Tensor = None):
        x = x + self.attention(self.ln1(x), mask)
        x = x + self.mlp(self.ln2(x))
        return x


class GPTransformer(torch.nn.Module):
    def __init__(self, n_blocks: int = 8, blocksize: int = 256, n_head: int = 16, edim: int = 2048, dropout: float = 0.0, bias: bool = False, causal: bool = False):
        super(GPTransformer, self).__init__()
        self.n_blocks = n_blocks
        self.transformer = torch.nn.ModuleList([
            Block(edim, n_head, dropout, bias, causal, blocksize) for _ in range(n_blocks)
        ])

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None):
        for block in self.transformer:
            x = block(x, mask)
        return x