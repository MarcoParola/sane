import hydra
import torch
import wandb
from src.models.sane.sane import Sane
from src.datasets.weights.tokenized_model_weights import TokenizedModelWeightDataset, TokenizedZooDataset
from src.utils.tokenizer import Tokenizer
from test_classifier import test_classifier 
# from src.utils.plots import layers_histogram
from src.datasets.utils import load_dataset
from src.models.utils import load_model
from src.utils.log import get_loggers
# from wandb import Image as WBImage
from pathlib import Path
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.callbacks import EarlyStopping


@hydra.main(config_path="config", config_name="config", version_base=None)
def main(cfg):
    # Loading SANE configuration
    stride = cfg.transformer.blocksize // cfg.training.stride

    # Load Tokenized Model Weights Dataset
    tokenizer = Tokenizer(cfg.transformer.blocksize)
    # if zoo mode is selected, load zoo weights
    if cfg.experiment.zoo:
        mode = "zoo_trained"
        loss_to_monitor = "val_loss"
        best_filename = "sane-{epoch:02d}-{val_loss:.4f}"
        zoo_models_path = []
        if cfg.experiment.zoo_models == "tinyimagenet_resnet18":
            print("Loading tinyimagenet zoo models ...")
            tinyimagenet_path = "checkpoints/tiny-imagenet_resnet18_kaiming_uniform_subset"
            zoo_models_path.append(tinyimagenet_path)
            mode = mode + "_tinyimagenet_resnet18"
            train_indices = list(range(0,50))
            val_indices = list(range(50,61))
            test_indices = list(range(61,72))
        elif cfg.experiment.zoo_models == "cnn":
            print("Loading cnn zoo models ...")
            cnn_zoo_path = "checkpoints/tune_zoo_cifar10_uniform_small"
            zoo_models_path.append(cnn_zoo_path)
            mode = mode + "_cnn"
            train_indices = list(range(0,700))
            val_indices = list(range(700,850))
            test_indices = list(range(850,1000))
        
        print("Loading training models...")
        if cfg.experiment.mode == "base":
            mode = mode + "_base"
            train_set = TokenizedZooDataset(zoo_models_path, tokenizer, cfg.transformer.blocksize, stride=stride, split_indices=train_indices)
        elif cfg.experiment.mode == "augmented":
            print(f"Training on augmented zoo dataset with noise of {cfg.experiment.noise_percentage*100}%...")
            mode = mode + "_augmented_" + str(int(cfg.experiment.noise_percentage*100))
            train_set = TokenizedZooDataset(zoo_models_path, tokenizer, cfg.transformer.blocksize, stride=stride, split_indices=train_indices, noise_percentage=cfg.experiment.noise_percentage)
        print("Loading validation models...")
        val_set = TokenizedZooDataset(zoo_models_path, tokenizer, cfg.transformer.blocksize, stride=stride, split_indices=val_indices)
        print("Loading testing models...")
        test_set = TokenizedZooDataset(zoo_models_path, tokenizer, cfg.transformer.blocksize, stride=stride, split_indices=test_indices)
        trainloader = torch.utils.data.DataLoader(dataset=train_set, batch_size=cfg.training.batch_size, shuffle=True, num_workers=0, persistent_workers=False)
        valloader = torch.utils.data.DataLoader(dataset=val_set, batch_size=cfg.training.batch_size, shuffle=False, num_workers=0, persistent_workers=False)
        testloader = torch.utils.data.DataLoader(dataset=test_set, batch_size=cfg.training.batch_size, shuffle=False, num_workers=0, persistent_workers=False)
    # else, load single model weights
    else:
        print("Loading single resnet18_tinyimagenet model ...")
        mode = "single_model_trained"
        loss_to_monitor = "train_loss"
        best_filename = "sane-{epoch:02d}-{train_loss:.4f}"
        original_checkpoint = torch.load(Path(f"checkpoints/{cfg.checkpoint}.pt"), weights_only=False)
        #original_checkpoint_location = "checkpoints/tiny-imagenet_resnet18_kaiming_uniform_subset/NN_tune_trainable_dbca4_00100_100_seed=101_2022-08-25_03-53-17/checkpoint_000060/checkpoints"
        #original_checkpoint = torch.load(original_checkpoint_location, weights_only=False)
        trainset = TokenizedModelWeightDataset(original_checkpoint, tokenizer, cfg.transformer.blocksize, stride=stride)
        trainloader = torch.utils.data.DataLoader(dataset=trainset, batch_size=cfg.training.batch_size, shuffle=True, num_workers=cfg.training.num_workers, persistent_workers=True)

    log_run = f"sane_{cfg.experiment.mode}.{mode}_sws{cfg.training.stride}_b{cfg.transformer.n_blocks}_e{cfg.transformer.edim}_h{cfg.transformer.n_head}"
    run_group, run_name = log_run.split(".")
    callbacks = list()
    wandb.finish()
    loggers = get_loggers(cfg, run_group, run_name)

    print("Initializing SANE model ...")
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
        dirpath=Path(f"out/{mode}", "sane_model", *log_run.split("."), "best"),
        filename=best_filename,
        monitor=loss_to_monitor,
        mode="min",
        save_top_k=1
    )
    callbacks.append(checkpoint_callback)

    # Early stopping callback
    early_stopping_callback = EarlyStopping(
        monitor = loss_to_monitor,
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
    if cfg.experiment.zoo:
        trainer.fit(sane_model, train_dataloaders=trainloader, val_dataloaders=valloader)
    else:
        trainer.fit(sane_model, train_dataloaders=trainloader)

    # Reconstruction/Test
    if cfg.experiment.zoo:
        trainer.test(sane_model, dataloaders=testloader)
    else:
        #sane_checkpoint = torch.load("checkpoints/sane_model/sws4_b2_e256_h2/best/single_tiny55_sane-epoch=998-train_loss=0.0000.ckpt", weights_only=False)
        #sane_model.load_state_dict(sane_checkpoint['state_dict'])
        #original_checkpoint = torch.load("checkpoints/tiny-imagenet_resnet18_kaiming_uniform_subset/NN_tune_trainable_dbca4_00055_55_seed=56_2022-08-23_23-59-10/checkpoint_000060/checkpoints", weights_only=False)

        # classification task preparation
        model_name = cfg.model.name
        dataset_name = cfg.dataset.name
        n_classes = cfg[dataset_name].num_classes
        if (model_name == "vit" and dataset_name == "cifar10"):
            img_size = 224
        else:
            img_size = cfg[dataset_name].img_size
        data_dir = cfg.data_dir

        train, val, test, remapping = load_dataset(dataset_name, data_dir, img_size)
        classifier_network = load_model(model_name, dataset_name).to(cfg.training.device)

        print(f"Testing reconstruction for {model_name} on {dataset_name} ...")
        testset = TokenizedModelWeightDataset(original_checkpoint, tokenizer, cfg.transformer.blocksize)
        testloader = torch.utils.data.DataLoader(dataset=testset, batch_size=cfg.training.batch_size, shuffle=False, num_workers=cfg.training.num_workers, persistent_workers=True)
        trainer.test(sane_model, dataloaders=testloader)
        recontokens, positions, embeddings = sane_model.get_test_outputs()
        injected_checkpoint = tokenizer.detokenize(recontokens, positions, original_checkpoint)
        injected_checkpoint_location = Path(f"checkpoints/injections/single_model_trained/{model_name}_{dataset_name}")
        injected_checkpoint_location.mkdir(777, parents=True, exist_ok=True)
        injected_checkpoint_location = injected_checkpoint_location.joinpath(f"injected.pt")
        injected_checkpoint_location.unlink(missing_ok=True); torch.save(injected_checkpoint, injected_checkpoint_location)
        injected_checkpoint = torch.load(injected_checkpoint_location, weights_only=False)
        
        # classification task
        print("\nOriginal Model:")
        classifier_network.load_state_dict(original_checkpoint)
        if dataset_name == 'tinyimagenet':
            classifier_network.eval()
        original_metrics = test_classifier(classifier_network, test, n_classes, cfg.training.batch_size, cfg.training.device, remapping)
        print("Injected Model:")
        classifier_network.load_state_dict(injected_checkpoint)
        if dataset_name == 'tinyimagenet':
            classifier_network.eval()
        injected_metrics = test_classifier(classifier_network, test, n_classes, cfg.training.batch_size, cfg.training.device, remapping)

        # layer by layer histogram plotting
        if trainer.logger:
            print("\nLogging...")
            wandb_run = trainer.logger.experiment
            #for idx, layer, figure, mse in layers_histogram(original_checkpoint, injected_checkpoint):
            #    if idx != -1: 
            #        wandb_run.log({f"{idx}.{layer}/plot": WBImage(figure), f"MSEs/{idx}.{layer}": mse})
            #    else: wandb_run.log({f"Test/{layer}": WBImage(figure)}) # layer becomes the plot's title
            
            wandb_run.log({f"Test/Original_{k}": v[0] for k,v in original_metrics.todict().items()})
            wandb_run.log({f"Test/Injected_{k}": v[0] for k,v in injected_metrics.todict().items()})
            wandb_run.finish()

        return


if __name__ == "__main__":
    main()