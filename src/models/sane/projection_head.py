import torch

class ProjectionHead(torch.nn.Module):
    def __init__(self, latdim: int, ntokens: int, odim: int):
        super(ProjectionHead, self).__init__()
        self.head = torch.nn.Sequential(
            torch.nn.Linear(latdim*ntokens, odim, bias=False),
            torch.nn.LayerNorm(odim),
            torch.nn.ReLU(),
            torch.nn.Linear(odim, odim, bias=False),
            torch.nn.LayerNorm(odim),
            torch.nn.ReLU(),
        )
    
    def forward(self, z: torch.Tensor):
        z = z.view(z.shape[0], -1) # (Bs, Tn, Ed) into (Bs, Tn*Ed)
        return self.head(z)