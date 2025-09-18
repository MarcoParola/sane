import copy
import torch
from pathlib import Path
import hydra


class Tokenizer():
    def __init__(self, tokensize: int = 0, return_mask: bool = True, ignore_bn = False):
        '''
            @Args:
                - tokensize: the size of a single token, default = 0 => tokensize is inferred from model's maximal neuron size
                - ignore_bn: flag indicating whether or not the batch_norm-like layers should be included in the tokenization
                - return_mask: flag indicating whether or not the padding masks should be returned with the tokens 
        '''
        self.tokensize = tokensize
        self.ignore_bn = ignore_bn
        self.return_mask = return_mask

    def tokenize(self, checkpoint: dict[str, torch.Tensor]):
        '''
            Tokenizes the weights of a checkpoint into tokens

            Args:
                checkpoint: dictionary containing the weights of the model
            
            Returns:
                tokens: tensor containing the tokenized weights
        '''
        # Ensure all tensors in checkpoint are on the same device
        device = next(iter(checkpoint.values())).device
        checkpoint = {k: v.to(device) for k, v in checkpoint.items()}
        
        #init output
        tokens = []
        pos = []
        masks = []

        tokensize = self.find_tokensize(checkpoint)

        # GET TOKENS
        idx = 0
        for key in checkpoint.keys():
            # if ignore_bn is True skip batchnorm layers
            if ("bn" in key or "downsample.1" in key or "batchnorm" in key) and self.ignore_bn:
                continue

            # get weights of all layers
            if "weight" in key or "running_mean" in key or "running_var" in key:
                w: torch.Tensor = checkpoint[key]
                # flatten to a 2D tensor of shape (num_neurons, neuron_size)
                w = w.view(w.shape[0], -1)
                
                # if the layer has a bias, concatenate it to the weights
                if "weight" in key and key.replace("weight", "bias") in checkpoint:
                    b: torch.Tensor = checkpoint[key.replace("weight", "bias")]
                    w = torch.cat([w, b.unsqueeze(1)], dim=1)

                # determine positions
                a = w.shape[1] // tokensize # number of tokens
                b = w.shape[1] % tokensize # residual size
                token_factor = int(a)
                if b > 0:
                    token_factor += 1

                # get position of each layer of the token
                idx_layer = [[idx, jdx] for jdx in range(w.shape[0]) for _ in range(token_factor)]
                # increase layyer counter
                idx += 1
                # add to overall position
                pos.extend(idx_layer)

                # tokenize the weights
                # if b > 0, pad the weights
                if b > 0:
                    mask = torch.zeros(w.shape[0], tokensize * token_factor, device=device)
                    mask[:, :w.shape[1]] = torch.ones(w.shape, device=device)
                    wpad = torch.zeros(w.shape[0], tokensize * token_factor, device=device)
                    wpad[:, :w.shape[1]] = w
                    w = wpad
                else:
                    mask = torch.ones(w.shape[0], tokensize * token_factor, device=device)

                # reshape the weights and masks to have tokensize columns
                w = w.view(-1, tokensize)
                mask = mask.view(-1, tokensize).to(torch.bool)

                # add to output
                tokens.append(w)
                masks.append(mask)

        # POST-PROCESSING
        # concatenate all tokens and masks
        tokens = torch.cat(tokens, dim=0)
        masks = torch.cat(masks, dim=0)
        useful_tokens = masks.sum().item()

        print(f"Token sequence length: {len(tokens)}")
        print(f"Total number of tokens: {len(tokens)} x {tokensize} = {len(tokens) * tokensize}")
        print(f"Useful tokens (no padding): {useful_tokens} ({useful_tokens / masks.numel() * 100:.2f}%)")

        # convert position to a tuple of (ndx, idx, jdx)
        pos = [(ndx, idx, jdx) for ndx, (idx, jdx) in enumerate(pos)]
        pos = torch.tensor(pos, device=device)
        # if the maximum position is greater than 32767, convert to int32, otherwise to int16
        if pos.max() > 32767:
            pos = pos.to(torch.int)
        else:
            pos = pos.to(torch.int16)

        return (tokens, masks, pos) if self.return_mask else (tokens, pos)    

    def detokenize(self, tokens, pos, reference_checkpoint, ignore_bn=False):
        """
        Detokenizes the given tokens and positions back to a checkpoint
        
        Args:
            tokens: sequence of tokenized weights
            pos: sequence of token positions
            reference_checkpoint: reference checkpoint to detokenize into.
            ignore_bn: whether to ignore batch normalization layers.
        
        Returns:
            checkpoint: checkpoint with weights and biases
        """
        # make a copy of the reference checkpoint to prevent memory management issues
        checkpoint = copy.deepcopy(reference_checkpoint)
        idx = 0
        for key in checkpoint.keys():
            if ("bn" in key or "downsample.1" in key or "batchnorm" in key) and ignore_bn:
                continue
            
            if "weight" in key or "running_mean" in key or "running_var" in key:
                # get module shapes
                mod_shape = checkpoint[key].shape

                # get the number of tokens for this module
                idx_channel = torch.where(pos[:, 1] == idx)[0]
                w_t = torch.index_select(input=tokens, index=idx_channel, dim=0)
        
                # calculate the content length of the module
                contentlength = int(torch.prod(torch.tensor(mod_shape)) / mod_shape[0])

                # update the checkpoint with the detokenized weights
                checkpoint[key] = w_t.view(mod_shape[0], -1)[:, :contentlength].view(mod_shape)

                # Keep running var > 0
                if "running_var" in key: checkpoint[key] = checkpoint[key].clamp(min=0)

                # check for biases
                if "weight" in key and key.replace("weight", "bias") in checkpoint:
                    checkpoint[key.replace("weight", "bias")] = w_t.view(mod_shape[0], -1)[:, contentlength]

                # update counter
                idx += 1
                
        return checkpoint
    

    def checkpoint_equality(self, ckpt1: dict[str, torch.Tensor], ckpt2: dict[str, torch.Tensor]):
        """
        Checks if two checkpoints are equal
        
        Args:
            ckpt1: first checkpoint
            ckpt2: second checkpoint
        
        Returns:
            True if the checkpoints are equal, False otherwise
        """

        # Check if the keys of both checkpoints are the same
        if not all([k1 == k2 for k1, k2 in zip(ckpt1.keys(), ckpt2.keys())]):
            return False
        
        for key in ckpt1.keys():
            # if ignore_bn is True skip batchnorm layers
            if ("bn" in key or "downsample.1" in key or "batchnorm" in key) and self.ignore_bn:
                continue
            
            # check if the keys are weights or running means/vars
            if "weight" in key or "running_mean" in key or "running_var" in key:
                # check if the shapes are the same
                if ckpt1[key].shape != ckpt2[key].shape:
                    return False
                # check if the values are the same
                if torch.any((ckpt1[key] - ckpt2[key]).abs() > 0).item():
                    return False
            
            # check if the key is a bias
            if "weight" in key and key.replace("weight", "bias") in ckpt1:
                # get the new key for bias
                newkey = key.replace("weight", "bias")
                if ckpt1[newkey].shape != ckpt2[newkey].shape:
                    return False
                if torch.any((ckpt1[newkey] - ckpt2[newkey]).abs() > 0).item():
                    return False
        
        return True         
    
    def debug_by_layer(self, checkpoint: dict[str, torch.Tensor]):
        device = next(iter(checkpoint.values())).device
        checkpoint = {k: v.to(device) for k, v in checkpoint.items()}
        tokensize = self.find_tokensize(checkpoint)

        print(f"\n{'='*50}\nDebug Tokenizer (Layer-by-Layer)\n{'='*50}")
        for key in checkpoint.keys():
            if ("bn" in key or "downsample.1" in key or "batchnorm" in key) and self.ignore_bn:
                continue
            if "weight" in key or "running_mean" in key or "running_var" in key:
                # layer name
                print(f"\nLayer: {key}") 

                # number of parameters (weights + biases)
                w: torch.Tensor = checkpoint[key]
                print(f"- Original weights shape: {w.shape} ({w.numel()} parameters)")
                w_flat = w.view(w.shape[0], -1)  # [num_neurons, flattened_dim]
                neuron_size = w_flat.shape[1]
                if "weight" in key and key.replace("weight", "bias") in checkpoint:
                    b: torch.Tensor = checkpoint[key.replace("weight", "bias")]
                    print(f"- Bias shape: {b.shape} ({b.numel()} parameters)")
                    w_flat = torch.cat([w_flat, b.unsqueeze(1)], dim=1)
                    neuron_size += 1
                    print(f"- Combined weights+bias shape: {w_flat.shape}")

                # token shape
                num_full_tokens = neuron_size // tokensize
                remainder = neuron_size % tokensize
                num_tokens_per_neuron = num_full_tokens + (1 if remainder > 0 else 0)
                total_tokens = w_flat.shape[0] * num_tokens_per_neuron
                print(f"- Final token shape: ({total_tokens}, {tokensize})")
        
    def find_tokensize(self, checkpoint: dict[str, torch.Tensor]) -> int:
        tokensize = self.tokensize
        if tokensize == 0:
            for key in checkpoint.keys():
                # if ignore_bn is True skip batchnorm layers
                if ("bn" in key or "downsample.1" in key or "batchnorm" in key) and self.ignore_bn:
                    continue

                # get weights of all layers
                if "weight" in key:
                    tmp = checkpoint[key].shape

                # get running mean and var for batchnorm layers
                elif "running_mean" in key or "running_var" in key:
                    tmp = checkpoint[key].shape
                else:
                    continue

                # calculate the size of the weight
                tempsize = torch.prod(torch.tensor(tmp))/tmp[0] 

                # if the size of the weight is greater than the current tokensize because of biases, update it
                if key.replace("weight", "bias") in checkpoint:
                    tempsize += 1
                if tempsize > tokensize:
                    tokensize = tempsize
        
        print(f"\nTokensize: {int(tokensize)}")
        return int(tokensize)


@hydra.main(version_base=None, config_path="../../config", config_name="config")
def main(cfg):
    # Load checkpoint and ensure it's on the same device
    checkpoint_name = cfg.checkpoint
    checkpoint = torch.load(Path("checkpoints", f"{checkpoint_name}.pt"))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint = {k: v.to(device) for k, v in checkpoint.items()}
    
    print(f"\nCheckpoint loaded: {checkpoint_name}")

    # Initialize tokenizer
    tokenizer = Tokenizer()
    
    # Tokenize the checkpoint
    tokens, masks, pos = tokenizer.tokenize(checkpoint)

    # Layer-by-layer debug
    #tokenizer.debug_by_layer(checkpoint)

    # Detokenize the tokens back to a checkpoint
    #detokenized_checkpoint = tokenizer.detokenize(tokens, pos, checkpoint)
    # Check if the original and detokenized checkpoints are equal
    #if tokenizer.checkpoint_equality(checkpoint, detokenized_checkpoint):
    #    print("\nDetokenization successful, checkpoints are equal")
    #else:
    #   print("\nDetokenization failed, checkpoints are not equal")


if __name__ == "__main__":
    main()