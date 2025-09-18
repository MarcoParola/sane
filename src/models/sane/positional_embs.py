import torch

class PositionalEmbs(torch.nn.Module):
    def __init__(self, max_positions: list[int], embedding_dimension: int):
        super(PositionalEmbs, self).__init__()
        self.max_positions = max_positions; self.embedding_dimension = embedding_dimension
        if len(max_positions) == 2:
            self.pe1 = torch.nn.Embedding(max_positions[0], embedding_dimension // 2)
            self.pe2 = torch.nn.Embedding(max_positions[1], embedding_dimension // 2)
            self.pe3 = None
        elif len(max_positions) == 3:
            self.pe1 = torch.nn.Embedding(max_positions[0], embedding_dimension // 2)  # add 1 + 2
            self.pe2 = torch.nn.Embedding(max_positions[1], embedding_dimension // 2)  # add 1 + 2
            self.pe3 = torch.nn.Embedding(max_positions[2], embedding_dimension // 2)  # cat 1+2 & 3
    
    def forward(self, inputs: torch.Tensor, positions: torch.Tensor):
        assert inputs.ndim == 3, f"Expecting 3D-tensor as input but got f{inputs.ndim}"
        assert positions.shape[2] == len(self.max_positions), f"Positions should have {len(self.max_positions)} dimensions, got: {positions.shape[2]}"
        assert positions.shape[0] == inputs.shape[0] and positions.shape[1] == inputs.shape[1], f"positions and inputs should have same shapes along dimensions 0 and 1"

        pe1 = self.pe1(positions[:, :, 0])
        pe2 = self.pe2(positions[:, :, 1])
        posemb = [pe1, pe2]
        if self.pe3 is not None: pe3 = self.pe3(positions[:,:,2]); posemb = [pe1+pe2, pe3]
        posemb = torch.cat(posemb, dim=2)
        return inputs + posemb