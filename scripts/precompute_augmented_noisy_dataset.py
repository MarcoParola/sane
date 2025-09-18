import torch
import os
import hydra
from src.models.utils import load_model


def add_noise_to_layers(model, layer_names, noise_std=0.01):
    '''
    Adds Gaussian noise to the weights and biases of specified layers in the model.
    
    Args:
        model (torch.nn.Module): The model to which noise will be added.
        layer_names (list of str): List of layer names to which noise will be added.
        noise_std (float): Standard deviation of the Gaussian noise.
    
    Returns:
        noise_state_dict (dict): A state dictionary with noisy weights and biases for the specified layers.
    '''
    noised_state_dict = {}
    
    for name, layer in model.named_modules():
        if any(ln in name for ln in layer_names):
            if hasattr(layer, 'weight') and layer.weight is not None:
                weight_name = f"{name}.weight"
                noisy_weight = layer.weight.data + torch.randn_like(layer.weight) * noise_std
                noised_state_dict[weight_name] = noisy_weight
            if hasattr(layer, 'bias') and layer.bias is not None:
                bias_name = f"{name}.bias"
                noisy_bias = layer.bias.data + torch.randn_like(layer.bias) * noise_std
                noised_state_dict[bias_name] = noisy_bias
    
    return noised_state_dict


def generate_noisy_checkpoints(model_name, dataset, n_checkpoints, layer_names, noise_std=0.01):
    '''
    Generates multiple noisy checkpoints of the model by adding noise to specified layers.
    
    Args:
        model (torch.nn.Module): The original model.
        n_checkpoints (int): Number of noisy checkpoints to generate.
        layer_names (list of str): List of layer names to which noise will be added.
        noise_std (float): Standard deviation of the Gaussian noise.
    '''
    base_output_dir = f"checkpoints/noisy/{model_name}_{n_checkpoints}_{noise_std}"
    counter = 1
    output_dir = f"{base_output_dir}_{counter}"

    # Check if directory exists and increment counter
    while os.path.exists(output_dir):
        counter += 1
        output_dir = f"{base_output_dir}_{counter}"
    
    os.makedirs(output_dir, exist_ok=True)

    for i in range(n_checkpoints):
        model = load_model(model_name, dataset)
        noisy_state_dict = add_noise_to_layers(model, layer_names)
        checkpoint_path = os.path.join(output_dir, f"noisy_model_{i+1}.pt")
        torch.save(noisy_state_dict, checkpoint_path)
        print(f"Saved noisy checkpoint {i+1} to {checkpoint_path}")


@hydra.main(version_base=None, config_path="../config", config_name="config")
def main(cfg):
    n_checkpoints = cfg.augmentation.n_checkpoints
    layers_to_noise = cfg.augmentation.layers_to_noise
    noise_std = cfg.augmentation.noise_std
    generate_noisy_checkpoints(cfg.model.name, cfg.dataset.name, n_checkpoints, layers_to_noise, noise_std)

if __name__ == "__main__":
    main()