import torch
from src.models.sane.sane import Sane
import hydra
from lightning.pytorch import Trainer
from src.utils.log import get_loggers
from src.datasets.weights.tokenized_model_weights import TokenizedModelWeightDataset, TokenizedAlignedZooDataset
from src.utils.tokenizer import Tokenizer
from pathlib import Path
from test_classifier import test_classifier 
# from src.utils.plots import layers_histogram
from src.datasets.utils import load_dataset
from src.models.utils import load_model
from src.utils.log import get_loggers
# from wandb import Image as WBImage
import wandb
from lightning.pytorch.loggers import WandbLogger
from src.utils.weight_matching import permute_model_zoo
from src.utils.metrics import ClassificationMetrics, SparsificationMetrics


def reconstruct_and_test_model(cfg, original_checkpoint, tokenizer, trainer, sane_model, classifier_network, test, n_classes, batch_size, device, remapping, i, original_metrics):
    testset = TokenizedModelWeightDataset(original_checkpoint, tokenizer, cfg.transformer.blocksize)
    testloader = torch.utils.data.DataLoader(dataset=testset, batch_size=cfg.training.batch_size, shuffle=False, num_workers=cfg.training.num_workers, persistent_workers=True)
    trainer.test(sane_model, dataloaders=testloader)
    recontokens, positions, embeddings = sane_model.get_test_outputs()
    injected_checkpoint = tokenizer.detokenize(recontokens, positions, original_checkpoint, ignore_pos=True)
    injected_checkpoint_location = Path("checkpoints/injections/sparsified")
    injected_checkpoint_location.mkdir(777, parents=True, exist_ok=True)
    injected_checkpoint_location = injected_checkpoint_location.joinpath(f"injected_{i}.pt")
    injected_checkpoint_location.unlink(missing_ok=True); torch.save(injected_checkpoint, injected_checkpoint_location)
    injected_checkpoint = torch.load(injected_checkpoint_location, weights_only=False)

    # classification task
    print(f"Injected Model {i}:")
    classifier_network.load_state_dict(injected_checkpoint)
    classifier_network.eval()
    injected_metrics = test_classifier(classifier_network, test, n_classes, batch_size, device, remapping)

    if cfg.experiment.sparsification:
        sparsification_metrics = SparsificationMetrics(original_checkpoint, injected_checkpoint, original_metrics.accuracy(), injected_metrics.accuracy(), device)

    # layer by layer histogram plotting
    if trainer.logger:
        print("\nLogging...")
        wandb_run = trainer.logger.experiment
        #for idx, layer, figure, mse in layers_histogram(original_checkpoint, injected_checkpoint):
        #    if idx != -1: 
        #        wandb_run.log({f"{idx}.{layer}/plot": WBImage(figure), f"MSEs/{idx}.{layer}": mse})
        #    else: wandb_run.log({f"Test/{layer}": WBImage(figure)}) # layer becomes the plot's title
        if cfg.experiment.sparsification:
            wandb_run.log({"Original_Acc": original_metrics.accuracy()})
            wandb_run.log({"Injected_Acc": injected_metrics.accuracy()})
            wandb_run.log({"Acc_Retention": sparsification_metrics.accuracy_retention()})
            wandb_run.log({"Compression_Rate": sparsification_metrics.compression_ratio()})
        else:
            wandb_run.log({f"Test/Original_{k}": v[0] for k,v in original_metrics.todict().items()})
            wandb_run.log({f"Test/Injected_{k}": v[0] for k,v in injected_metrics.todict().items()})
            wandb_run.finish()

    if cfg.experiment.sparsification:
        return sparsification_metrics.accuracy_retention(), sparsification_metrics.compression_ratio(), original_metrics.accuracy(), injected_metrics.accuracy()
    else:
        return original_metrics.accuracy(), injected_metrics.accuracy()


@hydra.main(config_path="config", config_name="config", version_base=None)
def main(cfg):
    stride = cfg.transformer.blocksize // cfg.training.stride
    log_run = f"test_sane_{cfg.experiment.mode}.test_sws{cfg.training.stride}_b{cfg.transformer.n_blocks}_e{cfg.transformer.edim}_h{cfg.transformer.n_head}"
    run_group, run_name = log_run.split(".")
    callbacks = list()
    loggers = get_loggers(cfg, run_group, run_name)

    print("Loading Sane architecture...")
    sane_model = Sane(
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

    if cfg.experiment.zoo:
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
            print("\nTesting reconstruction error on classification task...")
            test_indices = list(range(860,880))
            sane_model.projection_head.head[0] = torch.nn.Linear(6144, 30, bias=False)  # adjust projection head for CNNs weights size

            # classification task preparation
            model_name = cfg.model.name
            dataset_name = cfg.dataset.name
            n_classes = cfg[dataset_name].num_classes
            if (model_name == "vit" and dataset_name == "cifar10"):
                img_size = 224
            else:
                img_size = cfg[dataset_name].img_size
            batch_size = 32 if model_name == "vit" else cfg.training.batch_size
            device = cfg.training.device
            data_dir = cfg.data_dir

            train, val, test, remapping = load_dataset(dataset_name, data_dir, model_name, img_size)

            print(f"\nModel: {model_name} \nDataset: {dataset_name}")
            classifier_network = load_model(model_name, dataset_name).to(device)

            print("\nTesting reconstruction and predictions")
            # Iterate on selected split indices
            counter = 0
            if cfg.experiment.sparsification:
                compression_rates = []
                accuracy_retentions = []
                original_accuracies = []
                injected_accuracies = []
            else:
                relative_errors = []
            for i in test_indices:
                counter += 1
                checkpoint = aligned_models[i]
                wandb.finish()  # ensure previous run is closed
                if cfg.experiment.mode == "augmented":
                    if cfg.experiment.sparsification:
                        wandb_logger = WandbLogger(project="test_sane_sparsified", name=f"augmentation_{cfg.experiment.noise_percentage*100}%_model_{i}")
                    else:
                        wandb_logger = WandbLogger(project="test_sane", name=f"augmentation_{cfg.experiment.noise_percentage*100}%_model_{i}")
                else:
                    if cfg.experiment.sparsification:
                        wandb_logger = WandbLogger(project="test_sane_sparsified", name=f"model_{i}")
                    else:
                        wandb_logger = WandbLogger(project="test_sane", name=f"model_{i}")

                trainer = Trainer(logger=wandb_logger)
                
                # test original model only once
                #if counter == 1:
                print("\nOriginal Model:")
                classifier_network.load_state_dict(checkpoint)
                classifier_network.eval()
                original_metrics = test_classifier(classifier_network, test, n_classes, batch_size, device, remapping)

                print(f"\nReconstructing model {i}")
                if cfg.experiment.sparsification:
                    current_acc_ret, current_compression_ratio, current_orig_acc, current_inj_acc = reconstruct_and_test_model(cfg, checkpoint, tokenizer, trainer, sane_model, classifier_network, test, n_classes, batch_size, device, remapping, i, original_metrics)
                    accuracy_retentions.append(current_acc_ret)
                    compression_rates.append(current_compression_ratio)
                    original_accuracies.append(current_orig_acc)
                    injected_accuracies.append(current_inj_acc)
                else:
                    original_acc, injected_acc = reconstruct_and_test_model(cfg, checkpoint, tokenizer, trainer, sane_model, classifier_network, test, n_classes, batch_size, device, remapping, i, original_metrics)
                    current_relative_error = ClassificationMetrics.relative_error(original_acc, injected_acc)
                    relative_errors.append(current_relative_error)
            
            if cfg.experiment.sparsification:
                print(f"\nAverage original accuracy: {(sum(original_accuracies) / len(original_accuracies))*100:.2f}%")
                print(f"\nAverage injected accuracy: {(sum(injected_accuracies) / len(injected_accuracies))*100:.2f}%")
                print(f"\nAverage accuracy retention: {(sum(accuracy_retentions) / len(accuracy_retentions))*100:.2f}%")
                print(f"\nAverage compression rate: {(sum(compression_rates) / len(compression_rates)):.2f}x")
            else:
                print(f"\nAverage relative error: {(sum(relative_errors) / len(relative_errors))*100:.2f}%")

        else:
            original_checkpoint = torch.load(Path(f"checkpoints/{cfg.checkpoint}.pt"), weights_only=False)
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
            if dataset_name == 'tinyimagenet' or model_name == 'cnn':
                classifier_network.eval()
            original_metrics = test_classifier(classifier_network, test, n_classes, cfg.training.batch_size, cfg.training.device, remapping)
            print("Injected Model:")
            classifier_network.load_state_dict(injected_checkpoint)
            if dataset_name == 'tinyimagenet' or model_name == 'cnn':
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


if __name__ == "__main__":
    main()