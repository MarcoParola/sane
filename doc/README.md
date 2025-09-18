# DOC

## Installation

To install the project, simply clone the repository and get the necessary dependencies. Then, create a new project on [Weights & Biases](https://wandb.ai/site). Log in and paste your API key when prompted.

(**Important**: update the W&B username in the `./config/config.yaml` file)
```sh
# clone repo
git clone https://github.com/MarcoParola/sane-base.git
cd sane-base
mkdir data

# Create virtual environment and install dependencies 
python -m venv env
. env/bin/activate
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
python -m pip install -r requirements.txt 

# Weights&Biases login 
wandb login 
```


## Project Structure
The main modules composing the overall train and test procedures are in `./src/`. Specifically, `./src/` organizes the models (both for downstream tasks and autoencoders) and the datasets (both image datasets and tokenized downstream task model datasets).
For the sake of notation let's refer with "checkpoint" the specific models (usually classifiers).


## Usage
- Test to verify that the function that loads data works (it loads data from cifar10 and cifar100)
    ```
    python -m src.datasets.utils
    ```

- Test to verify that the function that loads the models works (it loads resnet18 pretrained on cifar10 and cifar100)
    ```
    python -m src.models.utils
    ```

- Test to verify the classification performance
    ```
    python -m test_classifier
    ```


## Main training script
It trains a SANE model on tokenized model weights, reconstructs the checkpoint and evaluates its impact on the classification performance of the reconstructed model.
```
python -m train_sane
```


## Additional utility scripts
In the `./scripts/` folder there are all independent files not involved in the main workflow.

- `analyze_checkpoint_layers.py`: analyzes the layers of a model checkpoint and prints a summary of their counts and total parameters.
    ```
    python -m scripts.check_tokenizer_output.py
    ```

- `from_torch_model_to_ptcheckpoint.py`: loads a model based on the given configuration, creates a checkpoint and saves the model's state dictionary as a pt checkpoint file.
    ```
    python -m scripts.from_torch_model_to_ptcheckpoint.py
    ```

- `precompute_augmented_noisy_data.py`: generates multiple noisy model checkpoints by adding Gaussian noise to the weights and biases of a specified list of layers, saving each noise version in a dedicated directory. The latter is based on the model name, number of checkpoints and noise standard deviation, in the form:
    ```
    checkpoints/noisy/{model_name}_{n_checkpoints}_{noise_std}_{counter}
    ```
    where `counter` is automatically incremented to avoid overwriting existing directories.

    ```
    python -m scripts.precompute_augmented_noisy_data.py
    ```
