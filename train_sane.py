import hydra
import torch
from src.models.sane.sane import Sane
from src.datasets.weights.tokenized_model_weights import TokenizedModelWeightDataset, TokenizedZooDataset
from src.utils.tokenizer import Tokenizer
from test_classifier import test_classifier 
from src.utils.plots import layers_histogram
from src.datasets.utils import load_dataset
from src.models.utils import load_model
from src.utils.log import get_loggers
from wandb import Image as WBImage
from pathlib import Path
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch import Trainer


@hydra.main(config_path="config", config_name="config", version_base=None)
def main(cfg):
    # Loading SANE configuration
    stride = cfg.transformer.blocksize // cfg.training.stride
    log_run = f"sane_{cfg.experiment.mode}.sws{cfg.training.stride}_b{cfg.transformer.n_blocks}_e{cfg.transformer.edim}_h{cfg.transformer.n_head}"
    run_group, run_name = log_run.split(".")
    callbacks = list()
    loggers = get_loggers(cfg, run_group, run_name)

    # Load Tokenized Model Weights Dataset
    tokenizer = Tokenizer(cfg.transformer.blocksize)
    # if zoo mode is selected, load zoo weights
    if cfg.experiment.zoo:
        zoo_models_path = []
        tinyimagenet_path = "checkpoints/tiny-imagenet_resnet18_kaiming_uniform_subset"
        zoo_models_path.append(tinyimagenet_path)
        dataset = TokenizedZooDataset(zoo_models_path, tokenizer, cfg.transformer.blocksize, stride=stride)
        train_set, val_set, test_set = torch.utils.data.random_split(dataset, [int(0.7*len(dataset)), int(0.15*len(dataset)), len(dataset) - int(0.7*len(dataset)) - int(0.15*len(dataset))])
        trainloader = torch.utils.data.DataLoader(dataset=train_set, batch_size=cfg.training.batch_size, shuffle=True, num_workers=0, persistent_workers=False)
        valloader = torch.utils.data.DataLoader(dataset=val_set, batch_size=cfg.training.batch_size, shuffle=False, num_workers=0, persistent_workers=False)
        testloader = torch.utils.data.DataLoader(dataset=test_set, batch_size=cfg.training.batch_size, shuffle=False, num_workers=0, persistent_workers=False)
    # else, load single model weights
    else:
        original_checkpoint = torch.load(Path(f"checkpoints/{cfg.checkpoint}.pt"), weights_only=False)
        trainset = TokenizedModelWeightDataset(original_checkpoint, tokenizer, cfg.transformer.blocksize, stride=stride)
        trainloader = torch.utils.data.DataLoader(dataset=trainset, batch_size=cfg.training.batch_size, shuffle=True, num_workers=cfg.training.num_workers, persistent_workers=True)

    sane_model = Sane(
        conf = cfg,
        idim = cfg.transformer.blocksize,
        edim = cfg.transformer.edim,
        n_head=cfg.transformer.n_head,
        n_blocks=cfg.transformer.n_blocks,
        wsize=cfg.transformer.blocksize
    )

    # Checkpoint callback to save the best model
    checkpoint_callback = ModelCheckpoint(
        dirpath=Path("out/sane_model", *log_run.split("."), "best"),
        filename="sane-{epoch:02d}-{val_loss:.4f}",
        monitor="val_loss",
        mode="min",
        save_top_k=1
    )
    callbacks.append(checkpoint_callback)

    # Training
    trainer = Trainer(
        max_epochs = cfg.training.n_epochs,
        callbacks = callbacks,
        logger = loggers
    )
    if cfg.experiment.zoo:
        trainer.fit(sane_model, train_dataloaders=trainloader, val_dataloaders=valloader)
    else:
        trainer.fit(sane_model, train_dataloaders=trainloader)

    # Reconstruction/Test
    if cfg.experiment.zoo:
        trainer.test(sane_model, dataloaders=testloader)
    else:
        testset = TokenizedModelWeightDataset(original_checkpoint, tokenizer, cfg.transformer.blocksize)
        testloader = torch.utils.data.DataLoader(dataset=testset, batch_size=cfg.training.batch_size, shuffle=False, num_workers=cfg.training.num_workers, persistent_workers=True)
        trainer.test(sane_model, dataloaders=testloader)
        recontokens, positions, embeddings = sane_model.get_test_outputs()
        injected_checkpoint = tokenizer.detokenize(recontokens, positions, original_checkpoint)
        store_location = Path("out", *log_run.split(".")) if log_run is not None else Path("out/sane")
        store_location.mkdir(777, parents=True, exist_ok=True)
        injected_checkpoint_location = store_location.joinpath("injections")
        injected_checkpoint_location.mkdir(777, parents=True, exist_ok=True)
        injected_checkpoint_location = injected_checkpoint_location.joinpath("injected.pt")
        injected_checkpoint_location.unlink(missing_ok=True); torch.save(injected_checkpoint, injected_checkpoint_location)
        
        # classification task preparation
        model_name = cfg.model.name
        dataset_name = cfg.dataset.name
        n_classes = cfg[dataset_name].num_classes
        if (model_name == "vit" and dataset_name == "cifar10"):
            img_size = 224
        else:
            img_size = cfg[dataset_name].img_size
        data_dir = cfg.data_dir

        train, val, test = load_dataset(dataset_name, data_dir, img_size)
        classifier_network = load_model(model_name, dataset_name).to(cfg.training.device)
        injected_checkpoint = torch.load(injected_checkpoint_location, weights_only=False)

        # classification task
        print("\nOriginal Model:")
        classifier_network.load_state_dict(original_checkpoint)
        original_metrics = test_classifier(classifier_network, test, n_classes, cfg.training.batch_size, cfg.training.device)
        print("Injected Model:")
        classifier_network.load_state_dict(injected_checkpoint)
        injected_metrics = test_classifier(classifier_network, test, n_classes, cfg.training.batch_size, cfg.training.device)

        # layer by layer histogram plotting
        if trainer.logger:
            print("\nLogging...")
            wandb_run = trainer.logger.experiment
            for idx, layer, figure, mse in layers_histogram(original_checkpoint, injected_checkpoint):
                if idx != -1: 
                    wandb_run.log({f"{idx}.{layer}/plot": WBImage(figure), f"MSEs/{idx}.{layer}": mse})
                else: wandb_run.log({f"Test/{layer}": WBImage(figure)}) # layer becomes the plot's title
            
            wandb_run.log({f"Test/Original_{k}": v[0] for k,v in original_metrics.todict().items()})
            wandb_run.log({f"Test/Injected_{k}": v[0] for k,v in injected_metrics.todict().items()})
            wandb_run.finish()

        return


if __name__ == "__main__":
    main()