import torch
import numpy as np
from src.utils.tokenizer import Tokenizer
from pathlib import Path


tokendstype = lambda tokensize: np.dtype([
    ('tokens', np.float_, (tokensize,)), 
    ('masks', np.float_, (tokensize,)), 
    ('positions', np.int_, (3,))
])

class TokenizedModelWeightDataset(torch.utils.data.Dataset):
    '''
    Tokenizes the passed checkpoint by using the received Tokenizer instance. Sets window_size for the dataset and the stride.
    '''

    def __init__(
        self, checkpoint: dict[str, torch.Tensor], tokenizer: Tokenizer, window_size: int = 1, 
        stride: int = None, fix_window: str = "padding"
    ):
        '''
        ### Arguments
            - checkpoint: the checkpoint to be tokenized
            - tokenizer: the Tokenizer instance to calculate tokens from checkpoints
            - window_size: number of tokens to be considered in each sample
            - stride: as in convolutional layer, sets the sliding step of the window
            - fix_window: either "shift" or "padding". Allows to select a method for handling last incomplete window either
            by moving the last window starting point back, to match the window_size (shift) or by zero-padding the remaining part
        '''
        assert window_size > 0, f"{window_size} invalid as window size, at least 1 token must be present in the window"
        tokens, masks, positions = tokenizer.tokenize(checkpoint)
        self.tokens = tokens.detach().clone().cpu()
        self.masks = masks.detach().clone().cpu()
        self.positions = positions.detach().clone().cpu()
        assert self.tokens.shape[0] == self.masks.shape[0] and self.masks.shape[0] == self.positions.shape[0], f"Inconsistency in received checkpoint: tshape - {self.tokens.shape}, mshape - {self.masks.shape}, pshape - {self.positions.shape}"
        self.window_size = min(window_size, self.tokens.shape[0])
        self.stride = stride if stride is not None else self.window_size
        self.fixwindow = fix_window

    def __len__(self):
        nwindows = (self.tokens.shape[0] - self.window_size) // self.stride + 1
        
        # numero token in ultima finestra (per ora parametri settati per avere exceeding = 0)
        exceeding = (self.tokens.shape[0] - self.window_size) % self.stride
        
        return nwindows + (1 if exceeding else 0)
    
    def tdim(self) -> int: return self.tokens.shape[1]
    
    def __getitem__(self, index):
        window_start = index*self.stride
        window_end = min(window_start+self.window_size, self.tokens.shape[0])
        if window_end-window_start < self.window_size:
            if self.fixwindow == "shift": 
                window_start -= (self.window_size + window_start - window_end)
                window_start = max(0, window_start)
                window_end = window_start + self.window_size
                tk, mk, ps = (self.tokens[window_start:window_end, :], self.masks[window_start:window_end, :], self.positions[window_start:window_end, :])
                return tk,mk,ps
            if self.fixwindow == "padding":
                tk, mk, ps = (self.tokens[window_start:window_end, :], self.masks[window_start:window_end, :], self.positions[window_start:window_end, :])
                padding_needed = self.window_size - (window_end - window_start)
                last_util_index = ps[-1, 0].item(); fake_layer_index = ps[-1, 1]+1
                
                tkpad = torch.zeros(self.window_size, tk.shape[1], dtype=tk.dtype)
                mkpad = torch.zeros(self.window_size, mk.shape[1], dtype=mk.dtype)
                pspad = torch.tensor([[last_util_index+tdx+1, fake_layer_index, tdx] for tdx in range(padding_needed)], dtype=ps.dtype)

                tkpad[:tk.shape[0], :] = tk; mkpad[:mk.shape[0], :] = mk; pspad = torch.cat([ps, pspad], dim=0)
                return tkpad,mkpad,pspad
            raise NotImplemented(f"available methods for window fixing: padding, shift, received: {self.fixwindow}")
        tk, mk, ps = (self.tokens[window_start:window_end, :], self.masks[window_start:window_end, :], self.positions[window_start:window_end, :])
        return tk,mk,ps
    

class TokenizedZooDataset(torch.utils.data.Dataset):
    '''
    Tokenizes the passed checkpoint zoo by using the received Tokenizer instance. Sets window_size for the dataset and the stride.
    '''

    def __init__(
        self, zoo_models_path: list[str], tokenizer: Tokenizer, window_size: int = 1, 
        stride: int = None, fix_window: str = "padding"
    ):
        '''
        ### Arguments
            - zoo_models_path: list of paths to the checkpoints to be tokenized
            - tokenizer: the Tokenizer instance to calculate tokens from checkpoints
            - window_size: number of tokens to be considered in each sample
            - stride: as in convolutional layer, sets the sliding step of the window
            - fix_window: either "shift" or "padding". Allows to select a method for handling last incomplete window either
            by moving the last window starting point back, to match the window_size (shift) or by zero-padding the remaining part
        '''
        assert window_size > 0, f"{window_size} invalid as window size, at least 1 token must be present in the window"

        all_tokens, all_masks, all_positions = [], [], []

        for zoo in zoo_models_path:
            zoo_path = Path(zoo)
            for folder in zoo_path.iterdir():
                if folder.is_dir():
                    current_checkpoint_path = folder / "checkpoint_000060/checkpoints"
                    if current_checkpoint_path.exists():
                        checkpoint = torch.load(current_checkpoint_path)

                        tokens, masks, positions = tokenizer.tokenize(checkpoint)
                        self.tokens = tokens.detach().clone().cpu()
                        self.masks = masks.detach().clone().cpu()
                        self.positions = positions.detach().clone().cpu()
                        assert self.tokens.shape[0] == self.masks.shape[0] and self.masks.shape[0] == self.positions.shape[0], f"Inconsistency in received checkpoint: tshape - {self.tokens.shape}, mshape - {self.masks.shape}, pshape - {self.positions.shape}"
                        all_tokens.append(tokens)
                        all_masks.append(masks)
                        all_positions.append(positions)
                    
        self.tokens = torch.cat(all_tokens, dim=0)
        self.masks = torch.cat(all_masks, dim=0)
        self.positions = torch.cat(all_positions, dim=0)
    
        self.window_size = min(window_size, self.tokens.shape[0])
        self.stride = stride if stride is not None else self.window_size
        self.fixwindow = fix_window

    def __len__(self):
        nwindows = (self.tokens.shape[0] - self.window_size) // self.stride + 1
        
        # numero token in ultima finestra (per ora parametri settati per avere exceeding = 0)
        exceeding = (self.tokens.shape[0] - self.window_size) % self.stride
        
        return nwindows + (1 if exceeding else 0)
    
    def tdim(self) -> int: return self.all_tokens[0].shape[1]
    
    def __getitem__(self, index):
        window_start = index*self.stride
        window_end = min(window_start+self.window_size, self.tokens.shape[0])
        if window_end-window_start < self.window_size:
            if self.fixwindow == "shift": 
                window_start -= (self.window_size + window_start - window_end)
                window_start = max(0, window_start)
                window_end = window_start + self.window_size
                tk, mk, ps = (self.tokens[window_start:window_end, :], self.masks[window_start:window_end, :], self.positions[window_start:window_end, :])
                return tk,mk,ps
            if self.fixwindow == "padding":
                tk, mk, ps = (self.tokens[window_start:window_end, :], self.masks[window_start:window_end, :], self.positions[window_start:window_end, :])
                padding_needed = self.window_size - (window_end - window_start)
                last_util_index = ps[-1, 0].item(); fake_layer_index = ps[-1, 1]+1
                
                tkpad = torch.zeros(self.window_size, tk.shape[1], dtype=tk.dtype)
                mkpad = torch.zeros(self.window_size, mk.shape[1], dtype=mk.dtype)
                pspad = torch.tensor([[last_util_index+tdx+1, fake_layer_index, tdx] for tdx in range(padding_needed)], dtype=ps.dtype)

                tkpad[:tk.shape[0], :] = tk; mkpad[:mk.shape[0], :] = mk; pspad = torch.cat([ps, pspad], dim=0)
                return tkpad,mkpad,pspad
            raise NotImplemented(f"available methods for window fixing: padding, shift, received: {self.fixwindow}")
        tk, mk, ps = (self.tokens[window_start:window_end, :], self.masks[window_start:window_end, :], self.positions[window_start:window_end, :])
        return tk,mk,ps