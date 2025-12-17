import torch
from src.models.sane.sanemaskpredictor import SaneMaskPredictor
import hydra
from lightning.pytorch import Trainer
from src.utils.log import get_loggers
from src.datasets.weights.tokenized_model_weights import TokenizedModelWeightDataset, TokenizedAlignedZooDataset
from src.utils.tokenizer import Tokenizer
from pathlib import Path 
from src.models.utils import load_model
import wandb
from lightning.pytorch.loggers import WandbLogger
from src.utils.weight_matching import permute_model_zoo

def generate_test_mask(cfg, original_checkpoint, tokenizer, trainer, sane_model, i):
    testset = TokenizedModelWeightDataset(original_checkpoint, tokenizer, cfg.transformer.blocksize)
    testloader = torch.utils.data.DataLoader(dataset=testset, batch_size=cfg.training.batch_size, shuffle=False, num_workers=cfg.training.num_workers, persistent_workers=True)
    trainer.test(sane_model, dataloaders=testloader)
    mask_logits = sane_model.get_test_masks()
    mask_probs = torch.sigmoid(mask_logits)
    binary_mask = (mask_probs > 0.5).float()

    # classification task
    print(f"Injected Model {i}:")

    total = binary_mask.numel()
    zeros = (binary_mask == 0).sum().item()
    sparsity = zeros / total * 100

    print(f"Sparsity: {sparsity:.2f}%")

    # layer by layer histogram plotting
    if trainer.logger:
        print("\nLogging...")
        wandb_run = trainer.logger.experiment
        wandb_run.log({"Sparsity": sparsity})

    return sparsity


@hydra.main(config_path="config", config_name="config", version_base=None)
def main(cfg):
    stride = cfg.transformer.blocksize // cfg.training.stride
    log_run = f"test_mask_predictor_{cfg.experiment.mode}.test"
    run_group, run_name = log_run.split(".")
    callbacks = list()
    loggers = get_loggers(cfg, run_group, run_name)

    print("Loading Sane architecture...")
    sane_model = SaneMaskPredictor(
        conf = cfg,
        idim = cfg.transformer.blocksize,
        edim = cfg.transformer.edim,
        n_head=cfg.transformer.n_head,
        n_blocks=cfg.transformer.n_blocks,
        wsize=cfg.transformer.blocksize
    )

    trainer = Trainer(
        max_epochs = cfg.training.n_epochs,
        callbacks = callbacks,
        logger = loggers
    )

    print("Loading pretrained Sane weights...")
    sane_checkpoint = torch.load(cfg.test.sane_checkpoint_path, map_location="cpu", weights_only=False)
    sane_model.load_state_dict(sane_checkpoint['state_dict'])

    tokenizer = Tokenizer(cfg.transformer.blocksize)

    zoo_name = cfg.model.name + "_" + cfg.dataset.name
    zoo_path = cfg[zoo_name].zoo_path

    print(f"Loading {zoo_name} zoo models ...")

    print("Aligning the zoo models to the the canoincal base to resolve asimmetries...")
    aligned_models = permute_model_zoo(zoo_path, cfg.model.name, cfg.dataset.name)

    if cfg.test.test_error:
        print("\nTesting Sane model...")
        test_indices = list(range(850,1000))
        test_set = TokenizedAlignedZooDataset(aligned_models, tokenizer, cfg.transformer.blocksize, stride=stride, split_indices=test_indices)
        testloader = torch.utils.data.DataLoader(dataset=test_set, batch_size=cfg.training.batch_size, shuffle=False, num_workers=0, persistent_workers=False)
        trainer.test(sane_model, dataloaders=testloader)

    if cfg.test.reconstruction_error:
        print("\nTesting mask generation task...")
        test_indices = list(range(860,880))
        #sane_model.projection_head.head[0] = torch.nn.Linear(6144, 30, bias=False)  # adjust projection head for CNNs weights size

        # classification task preparation
        model_name = cfg.model.name
        dataset_name = cfg.dataset.name
        device = cfg.training.device

        print(f"\nModel: {model_name} \nDataset: {dataset_name}")
        classifier_network = load_model(model_name, dataset_name).to(device)

        print("\nTesting reconstruction and predictions")
        # Iterate on selected split indices
        counter = 0
        sparsities = []
        for i in test_indices:
            counter += 1
            checkpoint = aligned_models[i]
            wandb.finish()  # ensure previous run is closed
            if cfg.experiment.mode == "augmented":
                wandb_logger = WandbLogger(project="test_mask", name=f"augmentation_{cfg.experiment.noise_percentage*100}%_model_{i}")
            else:
                wandb_logger = WandbLogger(project="test_mask", name=f"model_{i}")

            trainer = Trainer(logger=wandb_logger)
            
            # test original model only once
            #if counter == 1:
            print("\nOriginal Model:")
            classifier_network.load_state_dict(checkpoint)
            classifier_network.eval()

            print(f"\nReconstructing model {i}")
            sparsity = generate_test_mask(cfg, checkpoint, tokenizer, trainer, sane_model, i)
            sparsities.append(sparsity)
        
        print(f"\nAverage relative error: {(sum(sparsities) / len(sparsities))*100:.2f}%")



if __name__ == "__main__":
    main()