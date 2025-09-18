import torch
import hydra
from pathlib import Path
from src.utils.tokenizer import Tokenizer

def analyze_checkpoint_layers(checkpoint):
    """
    Analyze the checkpoint layers and print the number of parameters in each type of layer.
    """
    total_params = 0
    bn_params = 0
    conv_params = 0
    linear_params = 0
    bn_layers = 0
    conv_layers = 0
    linear_layers = 0
    ln_layers = 0
    ln_params = 0

    for key, value in checkpoint.items():
        if "weight" in key or "bias" in key or "running_mean" in key or "running_var" in key:
            total_params += value.numel()
            # Check convolutional parameters
            if "conv" in key or "downsample.0" in key or "patch_embed" in key:
                conv_layers += 1 if "bias" not in key else 0
                conv_params += value.numel()
                print(f"Layer: {key}")
            # Check batch normalization parameters
            elif "bn" in key or "batchnorm" in key or "downsample.1" in key:
                bn_layers += 1 if "bias" not in key else 0
                bn_params += value.numel()
            # Check linear parameters
            elif "fc" in key or "classifier" in key or "head" in key or "attn" in key:
                linear_layers += 1 if "bias" not in key else 0
                linear_params += value.numel()
            elif "norm" in key:
                ln_layers += 1 if "bias" not in key else 0
                ln_params += value.numel()


    print(f"Total number of parameters in the model: {total_params}")
    print(f"Number of parameters in {conv_layers} Convolutional layers: {conv_params}")
    print(f"Number of parameters in {bn_layers} BatchNorm layers: {bn_params}")
    print(f"Number of parameters in {ln_layers} LayerNorm layers: {ln_params}")
    print(f"Number of parameters in {linear_layers} Linear layers: {linear_params}")


@hydra.main(version_base=None, config_path="../config", config_name="config")
def main(cfg):
    # Load checkpoint and ensure it's on the same device
    checkpoint_name = cfg.checkpoint
    checkpoint = torch.load(Path("checkpoints", f"{checkpoint_name}.pt"))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint = {k: v.to(device) for k, v in checkpoint.items()}
    print(f"\nCheckpoint loaded: {checkpoint_name}")

    analyze_checkpoint_layers(checkpoint)

    # Initialize the tokenizer
    tokenizer = Tokenizer()
    
    # Tokenize the checkpoint
    print("\nTokenizing the checkpoint...")
    tokenizer.tokenize(checkpoint)


if __name__ == "__main__":
    main()