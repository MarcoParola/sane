import hydra
import torch
import wandb
from src.models.sane.sanemaskpredictor import CustomSparsitySaneMaskPredictor
from src.datasets.weights.tokenized_model_weights import CustomSparsitySparsifiedZooDataset
from src.utils.tokenizer import Tokenizer
from src.utils.log import get_loggers
from pathlib import Path
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.callbacks import EarlyStopping
from src.utils.weight_matching import permute_original_and_sparsified_zoo


@hydra.main(config_path="config", config_name="config", version_base=None)
def main(cfg):
    # Loading SANE configuration
    stride = cfg.transformer.blocksize // cfg.training.stride

    # Load Tokenized Model Weights Dataset
    tokenizer = Tokenizer(cfg.transformer.blocksize)

    zoo_name = cfg.model.name + "_" + cfg.dataset.name

    print(f"Loading {cfg.model.name}_{cfg.dataset.name} zoo models ...")
    zoo_path = cfg[zoo_name].zoo_path
    
    print(f"Loading {cfg.model.name}_{cfg.dataset.name} sparsified zoo models ...")
    pruners = ["random", "magnitude", "l_obs", "woodfisher", "chita"]
    sparsities = cfg.experiment.sparsities
    mode = zoo_name

    for pruner in pruners:
        mode = mode + "_" + pruner
    for sparsity in sparsities:
        mode = mode + "_" + str(int(sparsity*100))
    
    all_orig_models = []
    all_spars_models = []
    all_sparsities = []
    for pruner in pruners:
        for sparsity in sparsities:
            print(f"Loading sparsified zoo ({pruner}, sparsity {str(sparsity)})")
            sparsified_zoo_path = Path("checkpoints/sparsified_zoos/" + zoo_name + f"_{pruner}_sparsity_{sparsity}")
            print("Aligning the both original and sparsified model zoo...")
            aligned_models, aligned_sparsified_models = permute_original_and_sparsified_zoo(zoo_path, sparsified_zoo_path, cfg.model.name, cfg.dataset.name)
            for orig, spars in zip(aligned_models, aligned_sparsified_models):
                all_orig_models.append(orig)
                all_spars_models.append(spars)
                all_sparsities.append(sparsity)

    n_times = len(pruners) * len(sparsities)

    train_indices = []
    val_indices = []
    test_indices = []

    for i in range(n_times):
        base = i * 1000
        train_indices.extend(range(base, base + 700))
        val_indices.extend(range(base + 700, base + 850))
        test_indices.extend(range(base + 850, base + 1000))


    print("Loading training models...")
    if cfg.experiment.mode == "base":
        mode = mode + "_base"
        train_set = CustomSparsitySparsifiedZooDataset(all_orig_models, all_spars_models, all_sparsities, tokenizer, cfg.transformer.blocksize, stride=stride, split_indices=train_indices)
    elif cfg.experiment.mode == "augmented":
        print(f"Training on augmented zoo dataset with noise of {cfg.experiment.noise_percentage*100}%...")
        mode = mode + "_augmented_" + str(int(cfg.experiment.noise_percentage*100))
        train_set = CustomSparsitySparsifiedZooDataset(all_orig_models, all_spars_models, all_sparsities, tokenizer, cfg.transformer.blocksize, stride=stride, split_indices=train_indices, noise_percentage=cfg.experiment.noise_percentage)
    print("Loading validation models...")
    val_set = CustomSparsitySparsifiedZooDataset(all_orig_models, all_spars_models, all_sparsities, tokenizer, cfg.transformer.blocksize, stride=stride, split_indices=val_indices)
    print("Loading testing models...")
    trainloader = torch.utils.data.DataLoader(dataset=train_set, batch_size=cfg.training.batch_size, shuffle=True, num_workers=0, persistent_workers=False)
    valloader = torch.utils.data.DataLoader(dataset=val_set, batch_size=cfg.training.batch_size, shuffle=False, num_workers=0, persistent_workers=False)

    log_run = f"custom_sparsity_maskpredictor.{mode}"
    run_group, run_name = log_run.split(".")
    callbacks = list()
    wandb.finish()
    loggers = get_loggers(cfg, run_group, run_name)

    print("Initializing SANE model ...")
    sane_model = CustomSparsitySaneMaskPredictor(
        conf = cfg,
        idim = cfg.transformer.blocksize,
        edim = cfg.transformer.edim,
        n_head=cfg.transformer.n_head,
        n_blocks=cfg.transformer.n_blocks,
        wsize=cfg.transformer.blocksize
    )

    # Checkpoint callback to save the best model
    checkpoint_callback = ModelCheckpoint(
        dirpath=Path(f"out/{mode}", *log_run.split("."), "best"),
        filename="custom_spars-maskpredictor-{epoch:02d}-{val_loss:.4f}",
        monitor="val_loss",
        mode="min",
        save_top_k=1
    )
    callbacks.append(checkpoint_callback)

    # Early stopping callback
    early_stopping_callback = EarlyStopping(
        monitor = "val_loss",
        patience = cfg.training.patience,
        mode = "min",
        verbose = True
    )
    callbacks.append(early_stopping_callback) 

    # Training
    trainer = Trainer(
        max_epochs = cfg.training.n_epochs,
        callbacks = callbacks,
        logger = loggers
    )
    trainer.fit(sane_model, train_dataloaders=trainloader, val_dataloaders=valloader)


if __name__ == "__main__":
    main()