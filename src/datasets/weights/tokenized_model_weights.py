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
        
        # number of tokens in the last window
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
        stride: int = None, fix_window: str = "padding", split_indices: list[int] = None,  noise_percentage: float = 0.0
    ):
        '''
        ### Arguments
            - zoo_models_path: list of paths to the checkpoints to be tokenized
            - tokenizer: the Tokenizer instance to calculate tokens from checkpoints
            - window_size: number of tokens to be considered in each sample
            - stride: as in convolutional layer, sets the sliding step of the window
            - fix_window: either "shift" or "padding". Allows to select a method for handling last incomplete window either
              by moving the last window starting point back, to match the window_size (shift) or by zero-padding the remaining part
            - split_indices: list of indices to select which models to consider from the zoo
            - noise_percentage: percentage of noise to be added to the tokens, to augment the dataset
        '''
        assert window_size > 0, f"{window_size} invalid as window size, at least 1 token must be present in the window"

        all_tokens, all_masks, all_positions = [], [], []

        for zoo in zoo_models_path:
            zoo_path = Path(zoo)
            # Collect all folders of the current zoo
            model_folders = []
            for folder in zoo_path.iterdir():
                if folder.is_dir():
                    model_folders.append(folder)
            # Iterate on selected split indices
            counter = 0
            for i in split_indices:
                folder = model_folders[i]
                current_checkpoint_path = folder / "checkpoint_000050/checkpoints"
                if current_checkpoint_path.exists():
                    checkpoint = torch.load(current_checkpoint_path, weights_only=False)
                    # Add noise to the checkpoint if noise_percentage > 0
                    if noise_percentage > 0.0:
                        for key, value in checkpoint.items():
                            if isinstance(value, torch.Tensor):
                                if value.is_floating_point():
                                    noise = (torch.rand_like(value) * 2 - 1) * noise_percentage * value
                                    checkpoint[key] = value + noise
                    
                    counter+=1
                    print(f"\rTokenized {counter} models", end='', flush=True)
                    tokens, masks, positions = tokenizer.tokenize(checkpoint)
                    self.tokens = tokens.detach().clone().cpu()
                    self.masks = masks.detach().clone().cpu()
                    self.positions = positions.detach().clone().cpu()
                    assert self.tokens.shape[0] == self.masks.shape[0] and self.masks.shape[0] == self.positions.shape[0], f"Inconsistency in received checkpoint: tshape - {self.tokens.shape}, mshape - {self.masks.shape}, pshape - {self.positions.shape}"
                    all_tokens.append(tokens)
                    all_masks.append(masks)
                    all_positions.append(positions)
            print()

        self.tokens = torch.cat(all_tokens, dim=0) 
        '''
        total_len = sum(t.shape[0] for t in all_tokens)
        dim = all_tokens[0].shape[1]
        tokens_mm = np.memmap('tokens.dat', dtype='float32', mode='w+', shape=(total_len, dim))
        offset = 0
        for t in all_tokens:
            size = t.shape[0]
            tokens_mm[offset:offset+size] = t.cpu().numpy()
            offset += size
        '''
        print("Concatenated all tokens")

        self.masks = torch.cat(all_masks, dim=0)
        '''
        total_len = sum(m.shape[0] for m in all_masks)
        dim = all_masks[0].shape[1]
        masks_mm = np.memmap('masks.dat', dtype='float32', mode='w+', shape=(total_len, dim))
        offset = 0
        for m in all_masks:
            size = m.shape[0]
            masks_mm[offset:offset+size] = m.cpu().numpy()
            offset += size
        '''
        print("Concatenated all masks")

        self.positions = torch.cat(all_positions, dim=0)
        '''
        total_len = sum(p.shape[0] for p in all_positions)
        dim = all_positions[0].shape[1]
        positions_mm = np.memmap('positions.dat', dtype='int32', mode='w+', shape=(total_len, dim))
        offset = 0
        for p in all_positions:
            size = p.shape[0]
            positions_mm[offset:offset+size] = p.cpu().numpy()
            offset += size
        '''
        print("Concatenated all positions")
    
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

                tkpad = tkpad.to(tk.device)
                mkpad = mkpad.to(mk.device)
                pspad = pspad.to(ps.device)

                tkpad[:tk.shape[0], :] = tk; mkpad[:mk.shape[0], :] = mk; pspad = torch.cat([ps, pspad], dim=0)
                return tkpad,mkpad,pspad
            raise NotImplemented(f"available methods for window fixing: padding, shift, received: {self.fixwindow}")
        tk, mk, ps = (self.tokens[window_start:window_end, :], self.masks[window_start:window_end, :], self.positions[window_start:window_end, :])
        #tk = torch.from_numpy(self.tokens_mm[window_start:window_end, :])
        #mk = torch.from_numpy(self.masks_mm[window_start:window_end, :])
        #ps = torch.from_numpy(self.positions_mm[window_start:window_end, :])
        return tk,mk,ps
    

class TokenizedAlignedZooDataset(torch.utils.data.Dataset):
    '''
    Tokenizes the passed checkpoint zoo by using the received Tokenizer instance. Sets window_size for the dataset and the stride.
    '''

    def __init__(
        self, aligned_zoo: list[dict[str, torch.Tensor]], tokenizer: Tokenizer, window_size: int = 1, 
        stride: int = None, fix_window: str = "padding", split_indices: list[int] = None,  noise_percentage: float = 0.0, positional_emb: bool = False
    ):
        '''
        ### Arguments
            - aligned_zoo: list of checkpoints to be tokenized, already aligned to the canonical base
            - tokenizer: the Tokenizer instance to calculate tokens from checkpoints
            - window_size: number of tokens to be considered in each sample
            - stride: as in convolutional layer, sets the sliding step of the window
            - fix_window: either "shift" or "padding". Allows to select a method for handling last incomplete window either
              by moving the last window starting point back, to match the window_size (shift) or by zero-padding the remaining part
            - split_indices: list of indices to select which models to consider from the zoo
            - noise_percentage: percentage of noise to be added to the tokens, to augment the dataset
            - positional_emb: whether to return positional embeddings along with tokens and masks
        '''
        assert window_size > 0, f"{window_size} invalid as window size, at least 1 token must be present in the window"

        all_tokens, all_masks, all_positions = [], [], []
        counter = 0
        for i in split_indices:
            checkpoint = aligned_zoo[i]
            # Add noise to the checkpoint if noise_percentage > 0
            if noise_percentage > 0.0:
                for key, value in checkpoint.items():
                    if isinstance(value, torch.Tensor):
                        if value.is_floating_point():
                            noise = (torch.rand_like(value) * 2 - 1) * noise_percentage * value
                            checkpoint[key] = value + noise
            counter+=1
            print(f"\rTokenized {counter} model(s)", end='', flush=True)
            tokens, masks, positions = tokenizer.tokenize(checkpoint)
            self.tokens = tokens.detach().clone().cpu()
            self.masks = masks.detach().clone().cpu()
            self.positions = positions.detach().clone().cpu()
            assert self.tokens.shape[0] == self.masks.shape[0] and self.masks.shape[0] == self.positions.shape[0], f"Inconsistency in received checkpoint: tshape - {self.tokens.shape}, mshape - {self.masks.shape}, pshape - {self.positions.shape}"
            all_tokens.append(tokens)
            all_masks.append(masks)
            all_positions.append(positions)
        print()

        self.tokens = torch.cat(all_tokens, dim=0) 
        '''
        total_len = sum(t.shape[0] for t in all_tokens)
        dim = all_tokens[0].shape[1]
        tokens_mm = np.memmap('tokens.dat', dtype='float32', mode='w+', shape=(total_len, dim))
        offset = 0
        for t in all_tokens:
            size = t.shape[0]
            tokens_mm[offset:offset+size] = t.cpu().numpy()
            offset += size
        '''
        print("Concatenated all tokens")

        self.masks = torch.cat(all_masks, dim=0)
        '''
        total_len = sum(m.shape[0] for m in all_masks)
        dim = all_masks[0].shape[1]
        masks_mm = np.memmap('masks.dat', dtype='float32', mode='w+', shape=(total_len, dim))
        offset = 0
        for m in all_masks:
            size = m.shape[0]
            masks_mm[offset:offset+size] = m.cpu().numpy()
            offset += size
        '''
        print("Concatenated all masks")

        self.positions = torch.cat(all_positions, dim=0)
        '''
        total_len = sum(p.shape[0] for p in all_positions)
        dim = all_positions[0].shape[1]
        positions_mm = np.memmap('positions.dat', dtype='int32', mode='w+', shape=(total_len, dim))
        offset = 0
        for p in all_positions:
            size = p.shape[0]
            positions_mm[offset:offset+size] = p.cpu().numpy()
            offset += size
            '''
        if positional_emb:
            print("Concatenated all positions")
    
        self.window_size = min(window_size, self.tokens.shape[0])
        self.stride = stride if stride is not None else self.window_size
        self.fixwindow = fix_window
        self.positional_emb = positional_emb

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
                tk, mk = self.tokens[window_start:window_end, :], self.masks[window_start:window_end, :]
                if self.positional_emb:
                    ps = self.positions[window_start:window_end, :]
                else:
                    ps = torch.zeros_like(self.positions[window_start:window_end, :])
                return tk,mk,ps 
            if self.fixwindow == "padding":
                padding_needed = self.window_size - (window_end - window_start)
                
                tk, mk = self.tokens[window_start:window_end, :], self.masks[window_start:window_end, :]
                tkpad = torch.zeros(self.window_size, tk.shape[1], dtype=tk.dtype)
                mkpad = torch.zeros(self.window_size, mk.shape[1], dtype=mk.dtype)
                tkpad = tkpad.to(tk.device)
                mkpad = mkpad.to(mk.device)
                tkpad[:tk.shape[0], :] = tk; mkpad[:mk.shape[0], :] = mk; 

                if self.positional_emb:
                    ps = self.positions[window_start:window_end, :]
                    last_util_index = ps[-1, 0].item(); fake_layer_index = ps[-1, 1]+1
                    pspad = torch.tensor([[last_util_index+tdx+1, fake_layer_index, tdx] for tdx in range(padding_needed)], dtype=ps.dtype)
                    pspad = pspad.to(ps.device)
                    pspad = torch.cat([ps, pspad], dim=0)
                else:
                    # Return zeros of the same shape as the window
                    pspad = torch.zeros(self.window_size, 3, dtype=self.masks.dtype, device=self.masks.device)
                return tkpad,mkpad,pspad
            raise NotImplemented(f"available methods for window fixing: padding, shift, received: {self.fixwindow}")
        tk, mk = self.tokens[window_start:window_end, :], self.masks[window_start:window_end, :]
        #tk = torch.from_numpy(self.tokens_mm[window_start:window_end, :])
        #mk = torch.from_numpy(self.masks_mm[window_start:window_end, :])
        if self.positional_emb:
            ps = self.positions[window_start:window_end, :]
            #ps = torch.from_numpy(self.positions_mm[window_start:window_end, :])
        else:
            ps = torch.zeros_like(self.positions[window_start:window_end, :])
        return tk,mk,ps


class SparsifiedZooDataset(torch.utils.data.Dataset):
    '''
    Tokenizes the passed checkpoint zoo by using the received Tokenizer instance. Sets window_size for the dataset and the stride.
    '''

    def __init__(
        self, aligned_zoo: list[dict[str, torch.Tensor]], aligned_sparsified_zoo: list[dict[str, torch.Tensor]], tokenizer: Tokenizer, window_size: int = 1, 
        stride: int = None, fix_window: str = "padding", split_indices: list[int] = None,  noise_percentage: float = 0.0, positional_emb: bool = False
    ):
        '''
        ### Arguments
            - aligned_zoo: list of checkpoints to be tokenized, already aligned to the canonical base
            - aligned_sparsified_zoo: list of sparsified checkpoints, already aligned to the canonical base
            - tokenizer: the Tokenizer instance to calculate tokens from checkpoints
            - window_size: number of tokens to be considered in each sample
            - stride: as in convolutional layer, sets the sliding step of the window
            - fix_window: either "shift" or "padding". Allows to select a method for handling last incomplete window either
              by moving the last window starting point back, to match the window_size (shift) or by zero-padding the remaining part
            - split_indices: list of indices to select which models to consider from the zoo
            - noise_percentage: percentage of noise to be added to the tokens, to augment the dataset
            - positional_emb: whether to return positional embeddings along with tokens and masks
        '''
        assert window_size > 0, f"{window_size} invalid as window size, at least 1 token must be present in the window"

        all_tokens, all_masks, all_positions = [], [], []
        counter = 0
        for i in split_indices:
            checkpoint = aligned_zoo[i]
            # Add noise to the checkpoint if noise_percentage > 0
            if noise_percentage > 0.0:
                for key, value in checkpoint.items():
                    if isinstance(value, torch.Tensor):
                        if value.is_floating_point():
                            noise = (torch.rand_like(value) * 2 - 1) * noise_percentage * value
                            checkpoint[key] = value + noise
            counter+=1
            print(f"\rTokenized {counter} model(s)", end='', flush=True)
            tokens, masks, positions = tokenizer.tokenize(checkpoint)
            self.tokens = tokens.detach().clone().cpu()
            self.masks = masks.detach().clone().cpu()
            self.positions = positions.detach().clone().cpu()
            assert self.tokens.shape[0] == self.masks.shape[0] and self.masks.shape[0] == self.positions.shape[0], f"Inconsistency in received checkpoint: tshape - {self.tokens.shape}, mshape - {self.masks.shape}, pshape - {self.positions.shape}"
            all_tokens.append(tokens)
            all_masks.append(masks)
            all_positions.append(positions)
        print()

        self.tokens = torch.cat(all_tokens, dim=0) 
        '''
        total_len = sum(t.shape[0] for t in all_tokens)
        dim = all_tokens[0].shape[1]
        tokens_mm = np.memmap('tokens.dat', dtype='float32', mode='w+', shape=(total_len, dim))
        offset = 0
        for t in all_tokens:
            size = t.shape[0]
            tokens_mm[offset:offset+size] = t.cpu().numpy()
            offset += size
        '''
        print("Concatenated all tokens")

        self.masks = torch.cat(all_masks, dim=0)
        '''
        total_len = sum(m.shape[0] for m in all_masks)
        dim = all_masks[0].shape[1]
        masks_mm = np.memmap('masks.dat', dtype='float32', mode='w+', shape=(total_len, dim))
        offset = 0
        for m in all_masks:
            size = m.shape[0]
            masks_mm[offset:offset+size] = m.cpu().numpy()
            offset += size
        '''
        print("Concatenated all masks")

        self.positions = torch.cat(all_positions, dim=0)
        '''
        total_len = sum(p.shape[0] for p in all_positions)
        dim = all_positions[0].shape[1]
        positions_mm = np.memmap('positions.dat', dtype='int32', mode='w+', shape=(total_len, dim))
        offset = 0
        for p in all_positions:
            size = p.shape[0]
            positions_mm[offset:offset+size] = p.cpu().numpy()
            offset += size
            '''
        if positional_emb:
            print("Concatenated all positions")
    
        self.window_size = min(window_size, self.tokens.shape[0])
        self.stride = stride if stride is not None else self.window_size
        self.fixwindow = fix_window
        self.positional_emb = positional_emb

        all_sparsified_tokens = []
        counter = 0
        for i in split_indices:
            checkpoint = aligned_sparsified_zoo[i]
            # Add noise to the checkpoint if noise_percentage > 0
            if noise_percentage > 0.0:
                for key, value in checkpoint.items():
                    if isinstance(value, torch.Tensor):
                        if value.is_floating_point():
                            noise = (torch.rand_like(value) * 2 - 1) * noise_percentage * value
                            checkpoint[key] = value + noise
            counter+=1
            print(f"\rTokenized {counter} sparsified model(s)", end='', flush=True)
            sparsified_tokens, sparsified_masks, sparsified_positions = tokenizer.tokenize(checkpoint)
            self.sparsified_tokens = sparsified_tokens.detach().clone().cpu()
            self.sparsified_masks = sparsified_masks.detach().clone().cpu()
            self.sparsified_positions = sparsified_positions.detach().clone().cpu()
            assert self.sparsified_tokens.shape[0] == self.sparsified_masks.shape[0] and self.sparsified_masks.shape[0] == self.sparsified_positions.shape[0], f"Inconsistency in received checkpoint: tshape - {self.tokens.shape}, mshape - {self.masks.shape}, pshape - {self.positions.shape}"
            all_sparsified_tokens.append(sparsified_tokens)
        print()

        self.sparsified_tokens = torch.cat(all_sparsified_tokens, dim=0) 
        print("Concatenated all sparsified tokens")


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
                tk, mk = self.tokens[window_start:window_end, :], self.masks[window_start:window_end, :]
                # sparsified tokens
                stk = self.sparsified_tokens[window_start:window_end, :]
                if self.positional_emb:
                    ps = self.positions[window_start:window_end, :]
                else:
                    ps = torch.zeros_like(self.positions[window_start:window_end, :])
                return tk,stk,mk,ps 
            if self.fixwindow == "padding":
                padding_needed = self.window_size - (window_end - window_start)
                
                tk, mk = self.tokens[window_start:window_end, :], self.masks[window_start:window_end, :]
                stk = self.sparsified_tokens[window_start:window_end, :]
                tkpad = torch.zeros(self.window_size, tk.shape[1], dtype=tk.dtype)
                mkpad = torch.zeros(self.window_size, mk.shape[1], dtype=mk.dtype)
                stkpad = torch.zeros(self.window_size, stk.shape[1], dtype=stk.dtype)
                tkpad = tkpad.to(tk.device)
                mkpad = mkpad.to(mk.device)
                stkpad = stkpad.to(stk.device)
                tkpad[:tk.shape[0], :] = tk; mkpad[:mk.shape[0], :] = mk; 
                stkpad[:stk.shape[0], :] = stk

                if self.positional_emb:
                    ps = self.positions[window_start:window_end, :]
                    last_util_index = ps[-1, 0].item(); fake_layer_index = ps[-1, 1]+1
                    pspad = torch.tensor([[last_util_index+tdx+1, fake_layer_index, tdx] for tdx in range(padding_needed)], dtype=ps.dtype)
                    pspad = pspad.to(ps.device)
                    pspad = torch.cat([ps, pspad], dim=0)
                else:
                    # Return zeros of the same shape as the window
                    pspad = torch.zeros(self.window_size, 3, dtype=self.masks.dtype, device=self.masks.device)
                return tkpad,stkpad,mkpad,pspad
            raise NotImplemented(f"available methods for window fixing: padding, shift, received: {self.fixwindow}")
        tk, mk = self.tokens[window_start:window_end, :], self.masks[window_start:window_end, :]
        stk = self.sparsified_tokens[window_start:window_end, :]
        if self.positional_emb:
            ps = self.positions[window_start:window_end, :]
        else:
            ps = torch.zeros_like(self.positions[window_start:window_end, :])
        return tk,stk,mk,ps


class CustomSparsitySparsifiedZooDataset(torch.utils.data.Dataset):
    '''
    Tokenizes the passed checkpoint zoo by using the received Tokenizer instance. Sets window_size for the dataset and the stride.
    '''

    def __init__(
        self, aligned_zoo: list[dict[str, torch.Tensor]], aligned_sparsified_zoo: list[dict[str, torch.Tensor]], sparsities: list[float], tokenizer: Tokenizer, window_size: int = 1, 
        stride: int = None, fix_window: str = "padding", split_indices: list[int] = None,  noise_percentage: float = 0.0, positional_emb: bool = False
    ):
        '''
        ### Arguments
            - aligned_zoo: list of checkpoints to be tokenized, already aligned to the canonical base
            - aligned_sparsified_zoo: list of sparsified checkpoints, already aligned to the canonical base
            - sparsities: list of sparsity values corresponding to each checkpoint
            - tokenizer: the Tokenizer instance to calculate tokens from checkpoints
            - window_size: number of tokens to be considered in each sample
            - stride: as in convolutional layer, sets the sliding step of the window
            - fix_window: either "shift" or "padding". Allows to select a method for handling last incomplete window either
              by moving the last window starting point back, to match the window_size (shift) or by zero-padding the remaining part
            - split_indices: list of indices to select which models to consider from the zoo
            - noise_percentage: percentage of noise to be added to the tokens, to augment the dataset
            - positional_emb: whether to return positional embeddings along with tokens and masks
        '''
        assert window_size > 0, f"{window_size} invalid as window size, at least 1 token must be present in the window"

        all_tokens, all_masks, all_positions = [], [], []
        counter = 0
        for i in split_indices:
            checkpoint = aligned_zoo[i]
            # Add noise to the checkpoint if noise_percentage > 0
            if noise_percentage > 0.0:
                for key, value in checkpoint.items():
                    if isinstance(value, torch.Tensor):
                        if value.is_floating_point():
                            noise = (torch.rand_like(value) * 2 - 1) * noise_percentage * value
                            checkpoint[key] = value + noise
            counter+=1
            print(f"\rTokenized {counter} model(s)", end='', flush=True)
            tokens, masks, positions = tokenizer.tokenize(checkpoint)
            self.tokens = tokens.detach().clone().cpu()
            self.masks = masks.detach().clone().cpu()
            self.positions = positions.detach().clone().cpu()
            assert self.tokens.shape[0] == self.masks.shape[0] and self.masks.shape[0] == self.positions.shape[0], f"Inconsistency in received checkpoint: tshape - {self.tokens.shape}, mshape - {self.masks.shape}, pshape - {self.positions.shape}"
            tokensize = tokens.shape[1]
            sparsity_token = torch.full((1, tokensize), sparsities[i], dtype=tokens.dtype)
            sparsity_mask = torch.zeros((1, tokensize), dtype=masks.dtype)
            sparsity_pos = torch.zeros((1, 3), dtype=positions.dtype)
            all_tokens.append(sparsity_token)
            all_masks.append(sparsity_mask)
            all_positions.append(sparsity_pos)
            all_tokens.append(tokens)
            all_masks.append(masks)
            all_positions.append(positions)
        print()

        self.tokens = torch.cat(all_tokens, dim=0) 
        print("Concatenated all tokens")

        self.masks = torch.cat(all_masks, dim=0)
        print("Concatenated all masks")

        self.positions = torch.cat(all_positions, dim=0)
        if positional_emb:
            print("Concatenated all positions")
    
        self.window_size = min(window_size, self.tokens.shape[0])
        self.stride = stride if stride is not None else self.window_size
        self.fixwindow = fix_window
        self.positional_emb = positional_emb

        all_sparsified_tokens = []
        counter = 0
        for i in split_indices:
            checkpoint = aligned_sparsified_zoo[i]
            # Add noise to the checkpoint if noise_percentage > 0
            if noise_percentage > 0.0:
                for key, value in checkpoint.items():
                    if isinstance(value, torch.Tensor):
                        if value.is_floating_point():
                            noise = (torch.rand_like(value) * 2 - 1) * noise_percentage * value
                            checkpoint[key] = value + noise
            counter+=1
            print(f"\rTokenized {counter} sparsified model(s)", end='', flush=True)
            sparsified_tokens, sparsified_masks, sparsified_positions = tokenizer.tokenize(checkpoint)
            self.sparsified_tokens = sparsified_tokens.detach().clone().cpu()
            self.sparsified_masks = sparsified_masks.detach().clone().cpu()
            self.sparsified_positions = sparsified_positions.detach().clone().cpu()
            assert self.sparsified_tokens.shape[0] == self.sparsified_masks.shape[0] and self.sparsified_masks.shape[0] == self.sparsified_positions.shape[0], f"Inconsistency in received checkpoint: tshape - {self.tokens.shape}, mshape - {self.masks.shape}, pshape - {self.positions.shape}"
            sparsity_token = torch.full((1, tokensize), sparsities[i], dtype=sparsified_tokens.dtype)
            all_sparsified_tokens.append(sparsity_token)
            all_sparsified_tokens.append(sparsified_tokens)
        print()

        self.sparsified_tokens = torch.cat(all_sparsified_tokens, dim=0) 
        print("Concatenated all sparsified tokens")


    def __len__(self):
        nwindows = (self.tokens.shape[0] - self.window_size) // self.stride + 1
        
        # numero token in ultima finestra (per ora parametri settati per avere exceeding = 0)
        exceeding = (self.tokens.shape[0] - self.window_size) % self.stride
        
        return nwindows + (1 if exceeding else 0)
    
    def tdim(self) -> int: return self.all_tokens[1].shape[1]
    
    def __getitem__(self, index):
        window_start = index*self.stride
        window_end = min(window_start+self.window_size, self.tokens.shape[0])
        if window_end-window_start < self.window_size:
            if self.fixwindow == "shift": 
                window_start -= (self.window_size + window_start - window_end)
                window_start = max(0, window_start)
                window_end = window_start + self.window_size
                tk, mk = self.tokens[window_start:window_end, :], self.masks[window_start:window_end, :]
                # sparsified tokens
                stk = self.sparsified_tokens[window_start:window_end, :]
                if self.positional_emb:
                    ps = self.positions[window_start:window_end, :]
                else:
                    ps = torch.zeros_like(self.positions[window_start:window_end, :])
                return tk,stk,mk,ps 
            if self.fixwindow == "padding":
                padding_needed = self.window_size - (window_end - window_start)
                
                tk, mk = self.tokens[window_start:window_end, :], self.masks[window_start:window_end, :]
                stk = self.sparsified_tokens[window_start:window_end, :]
                tkpad = torch.zeros(self.window_size, tk.shape[1], dtype=tk.dtype)
                mkpad = torch.zeros(self.window_size, mk.shape[1], dtype=mk.dtype)
                stkpad = torch.zeros(self.window_size, stk.shape[1], dtype=stk.dtype)
                tkpad = tkpad.to(tk.device)
                mkpad = mkpad.to(mk.device)
                stkpad = stkpad.to(stk.device)
                tkpad[:tk.shape[0], :] = tk; mkpad[:mk.shape[0], :] = mk; 
                stkpad[:stk.shape[0], :] = stk

                if self.positional_emb:
                    ps = self.positions[window_start:window_end, :]
                    last_util_index = ps[-1, 0].item(); fake_layer_index = ps[-1, 1]+1
                    pspad = torch.tensor([[last_util_index+tdx+1, fake_layer_index, tdx] for tdx in range(padding_needed)], dtype=ps.dtype)
                    pspad = pspad.to(ps.device)
                    pspad = torch.cat([ps, pspad], dim=0)
                else:
                    # Return zeros of the same shape as the window
                    pspad = torch.zeros(self.window_size, 3, dtype=self.masks.dtype, device=self.masks.device)
                return tkpad,stkpad,mkpad,pspad
            raise NotImplemented(f"available methods for window fixing: padding, shift, received: {self.fixwindow}")
        tk, mk = self.tokens[window_start:window_end, :], self.masks[window_start:window_end, :]
        stk = self.sparsified_tokens[window_start:window_end, :]
        if self.positional_emb:
            ps = self.positions[window_start:window_end, :]
        else:
            ps = torch.zeros_like(self.positions[window_start:window_end, :])
        return tk,stk,mk,ps


class CustomSparsityZooDataset(torch.utils.data.Dataset):
    '''
    Tokenizes the passed checkpoint zoo by using the received Tokenizer instance. Sets window_size for the dataset and the stride.
    '''

    def __init__(
        self, aligned_zoo: list[dict[str, torch.Tensor]], sparsity: float, tokenizer: Tokenizer, window_size: int = 1, 
        stride: int = None, fix_window: str = "padding", split_indices: list[int] = None,  noise_percentage: float = 0.0, positional_emb: bool = False
    ):
        '''
        ### Arguments
            - aligned_zoo: list of checkpoints to be tokenized, already aligned to the canonical base
            - sparsity: sparsity value to be used for all checkpoints
            - tokenizer: the Tokenizer instance to calculate tokens from checkpoints
            - window_size: number of tokens to be considered in each sample
            - stride: as in convolutional layer, sets the sliding step of the window
            - fix_window: either "shift" or "padding". Allows to select a method for handling last incomplete window either
              by moving the last window starting point back, to match the window_size (shift) or by zero-padding the remaining part
            - split_indices: list of indices to select which models to consider from the zoo
            - noise_percentage: percentage of noise to be added to the tokens, to augment the dataset
            - positional_emb: whether to return positional embeddings along with tokens and masks
        '''
        assert window_size > 0, f"{window_size} invalid as window size, at least 1 token must be present in the window"

        all_tokens, all_masks, all_positions = [], [], []
        counter = 0
        for i in split_indices:
            checkpoint = aligned_zoo[i]
            # Add noise to the checkpoint if noise_percentage > 0
            if noise_percentage > 0.0:
                for key, value in checkpoint.items():
                    if isinstance(value, torch.Tensor):
                        if value.is_floating_point():
                            noise = (torch.rand_like(value) * 2 - 1) * noise_percentage * value
                            checkpoint[key] = value + noise
            counter+=1
            print(f"\rTokenized {counter} model(s)", end='', flush=True)
            tokens, masks, positions = tokenizer.tokenize(checkpoint)
            self.tokens = tokens.detach().clone().cpu()
            self.masks = masks.detach().clone().cpu()
            self.positions = positions.detach().clone().cpu()
            assert self.tokens.shape[0] == self.masks.shape[0] and self.masks.shape[0] == self.positions.shape[0], f"Inconsistency in received checkpoint: tshape - {self.tokens.shape}, mshape - {self.masks.shape}, pshape - {self.positions.shape}"
            tokensize = tokens.shape[1]
            sparsity_token = torch.full((1, tokensize), sparsity, dtype=tokens.dtype)
            sparsity_mask = torch.zeros((1, tokensize), dtype=masks.dtype)
            sparsity_pos = torch.zeros((1, 3), dtype=positions.dtype)
            all_tokens.append(sparsity_token)
            all_masks.append(sparsity_mask)
            all_positions.append(sparsity_pos)
            all_tokens.append(tokens)
            all_masks.append(masks)
            all_positions.append(positions)
        print()

        self.tokens = torch.cat(all_tokens, dim=0) 
        print("Concatenated all tokens")

        self.masks = torch.cat(all_masks, dim=0)
        print("Concatenated all masks")

        self.positions = torch.cat(all_positions, dim=0)
        if positional_emb:
            print("Concatenated all positions")
    
        self.window_size = min(window_size, self.tokens.shape[0])
        self.stride = stride if stride is not None else self.window_size
        self.fixwindow = fix_window
        self.positional_emb = positional_emb

    def __len__(self):
        nwindows = (self.tokens.shape[0] - self.window_size) // self.stride + 1
        
        # numero token in ultima finestra (per ora parametri settati per avere exceeding = 0)
        exceeding = (self.tokens.shape[0] - self.window_size) % self.stride
        
        return nwindows + (1 if exceeding else 0)
    
    def tdim(self) -> int: return self.all_tokens[1].shape[1]
    
    def __getitem__(self, index):
        window_start = index*self.stride
        window_end = min(window_start+self.window_size, self.tokens.shape[0])
        if window_end-window_start < self.window_size:
            if self.fixwindow == "shift": 
                window_start -= (self.window_size + window_start - window_end)
                window_start = max(0, window_start)
                window_end = window_start + self.window_size
                tk, mk = self.tokens[window_start:window_end, :], self.masks[window_start:window_end, :]
                if self.positional_emb:
                    ps = self.positions[window_start:window_end, :]
                else:
                    ps = torch.zeros_like(self.positions[window_start:window_end, :])
                return tk,mk,ps 
            if self.fixwindow == "padding":
                padding_needed = self.window_size - (window_end - window_start)
                
                tk, mk = self.tokens[window_start:window_end, :], self.masks[window_start:window_end, :]
                tkpad = torch.zeros(self.window_size, tk.shape[1], dtype=tk.dtype)
                mkpad = torch.zeros(self.window_size, mk.shape[1], dtype=mk.dtype)
                tkpad = tkpad.to(tk.device)
                mkpad = mkpad.to(mk.device)
                tkpad[:tk.shape[0], :] = tk; mkpad[:mk.shape[0], :] = mk; 

                if self.positional_emb:
                    ps = self.positions[window_start:window_end, :]
                    last_util_index = ps[-1, 0].item(); fake_layer_index = ps[-1, 1]+1
                    pspad = torch.tensor([[last_util_index+tdx+1, fake_layer_index, tdx] for tdx in range(padding_needed)], dtype=ps.dtype)
                    pspad = pspad.to(ps.device)
                    pspad = torch.cat([ps, pspad], dim=0)
                else:
                    # Return zeros of the same shape as the window
                    pspad = torch.zeros(self.window_size, 3, dtype=self.masks.dtype, device=self.masks.device)
                return tkpad,mkpad,pspad
            raise NotImplemented(f"available methods for window fixing: padding, shift, received: {self.fixwindow}")
        tk, mk = self.tokens[window_start:window_end, :], self.masks[window_start:window_end, :]
        #tk = torch.from_numpy(self.tokens_mm[window_start:window_end, :])
        #mk = torch.from_numpy(self.masks_mm[window_start:window_end, :])
        if self.positional_emb:
            ps = self.positions[window_start:window_end, :]
            #ps = torch.from_numpy(self.positions_mm[window_start:window_end, :])
        else:
            ps = torch.zeros_like(self.positions[window_start:window_end, :])
        return tk,mk,ps


class CustomSparsityModelDataset(torch.utils.data.Dataset):
    '''
    Tokenizes the passed checkpoint by using the received Tokenizer instance. Sets window_size for the dataset and the stride.
    '''

    def __init__(
        self, checkpoint: dict[str, torch.Tensor], sparsity: float, tokenizer: Tokenizer, window_size: int = 1, 
        stride: int = None, fix_window: str = "padding"
    ):
        '''
        ### Arguments
            - checkpoint: the checkpoint to be tokenized
            - sparsity: the sparsity value to be prepended to the tokens
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
        
        tokensize = tokens.shape[1]
        sparsity_token = torch.full((1, tokensize), sparsity, dtype=tokens.dtype)
        sparsity_mask = torch.zeros((1, tokensize), dtype=masks.dtype)
        sparsity_pos = torch.zeros((1, 3), dtype=positions.dtype)
        all_tokens = [sparsity_token, self.tokens]
        all_masks = [sparsity_mask, self.masks]
        all_positions = [sparsity_pos, self.positions]
        self.tokens = torch.cat(all_tokens, dim=0)
        self.masks = torch.cat(all_masks, dim=0)
        self.positions = torch.cat(all_positions, dim=0)

        self.window_size = min(window_size, self.tokens.shape[0])
        self.stride = stride if stride is not None else self.window_size
        self.fixwindow = fix_window

    def __len__(self):
        nwindows = (self.tokens.shape[0] - self.window_size) // self.stride + 1
        
        # number of tokens in the last window
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