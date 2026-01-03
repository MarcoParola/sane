import torch
from src.models.sane.sanemaskpredictor import CustomSparsitySaneMaskPredictor
import hydra
from lightning.pytorch import Trainer
from src.utils.log import get_loggers
from src.datasets.weights.tokenized_model_weights import CustomSparsityZooDataset, CustomSparsityModelDataset
from src.utils.tokenizer import Tokenizer
from src.models.utils import load_model
from src.datasets.utils import load_dataset
import wandb
from lightning.pytorch.loggers import WandbLogger
from src.utils.weight_matching import permute_model_zoo
from test_classifier import test_classifier
from src.utils.metrics import SparsificationMetrics

def generate_test_mask(cfg, original_checkpoint, tokenizer, trainer, sane_model, classifier_network, test, n_classes, batch_size, device, remapping, i, original_metrics, target_sparsity):
    testset = CustomSparsityModelDataset(original_checkpoint, target_sparsity, tokenizer, cfg.transformer.blocksize)
    testloader = torch.utils.data.DataLoader(dataset=testset, batch_size=cfg.training.batch_size, shuffle=False, num_workers=cfg.training.num_workers, persistent_workers=True)
    trainer.test(sane_model, dataloaders=testloader)
    
    mask_logits, tokens, positions = sane_model.get_test_outputs()
    mask_probs = torch.sigmoid(mask_logits[1:])
    binary_mask = (mask_probs > 0.5).float()

    print(f"Injected Model {i}:")

    total = binary_mask.numel()
    zeros = (binary_mask == 0).sum().item()
    sparsity = zeros / total * 100

    print(f"Sparsity: {sparsity:.2f}%")

    # apply mask to checkpoint
    assert tokens[1:].shape == binary_mask.shape
    masked_tokens = tokens[1:] * binary_mask

    masked_checkpoint = tokenizer.detokenize(masked_tokens, positions[1:], original_checkpoint, ignore_pos=True)
    classifier_network.load_state_dict(masked_checkpoint)
    classifier_network.eval()
    injected_metrics = test_classifier(classifier_network, test, n_classes, batch_size, device, remapping)

    sparsification_metrics = SparsificationMetrics(original_checkpoint, masked_checkpoint, original_metrics.accuracy(), injected_metrics.accuracy(), device)

    # layer by layer histogram plotting
    if trainer.logger:
        print("\nLogging...")
        wandb_run = trainer.logger.experiment
        wandb_run.log({"Original_Acc": original_metrics.accuracy()})
        wandb_run.log({"Injected_Acc": injected_metrics.accuracy()})
        wandb_run.log({"Acc_Retention": sparsification_metrics.accuracy_retention()})
        wandb_run.log({"Compression_Ratio": sparsification_metrics.compression_ratio()})
        wandb_run.log({"Sparsity": sparsity})

    return sparsification_metrics.accuracy_retention(), sparsification_metrics.compression_ratio(), original_metrics.accuracy(), injected_metrics.accuracy(), sparsity


@hydra.main(config_path="config", config_name="config", version_base=None)
def main(cfg):
    stride = cfg.transformer.blocksize // cfg.training.stride
    log_run = f"test_mask_predictor_{cfg.experiment.mode}.test"
    run_group, run_name = log_run.split(".")
    callbacks = list()
    loggers = get_loggers(cfg, run_group, run_name)

    print("Loading Sane architecture...")
    sane_model = CustomSparsitySaneMaskPredictor(
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
    target_sparsity = cfg.experiment.target_sparsity

    if cfg.test.test_error:
        print("\nTesting Sane model...")
        test_indices = list(range(850,1000))
        test_set = CustomSparsityZooDataset(aligned_models, target_sparsity, tokenizer, cfg.transformer.blocksize, stride=stride, split_indices=test_indices)
        testloader = torch.utils.data.DataLoader(dataset=test_set, batch_size=cfg.training.batch_size, shuffle=False, num_workers=0, persistent_workers=False)
        trainer.test(sane_model, dataloaders=testloader)

    if cfg.test.reconstruction_error:
        print("\nTesting mask generation task...")
        test_indices = list(range(860,880))

        # classification task preparation
        model_name = cfg.model.name
        dataset_name = cfg.dataset.name
        n_classes = cfg[dataset_name].num_classes
        img_size = cfg[dataset_name].img_size
        batch_size = cfg.training.batch_size
        device = cfg.training.device
        data_dir = cfg.data_dir

        train, val, test, remapping = load_dataset(dataset_name, data_dir, model_name, img_size)

        print(f"\nModel: {model_name} \nDataset: {dataset_name}")
        classifier_network = load_model(model_name, dataset_name).to(device)

        print("\nTesting reconstruction and predictions")
        # Iterate on selected split indices
        counter = 0
        compression_rates = []
        accuracy_retentions = []
        original_accuracies = []
        injected_accuracies = []
        sparsities = []
        for i in test_indices:
            counter += 1
            checkpoint = aligned_models[i]
            wandb.finish()  # ensure previous run is closed
            if cfg.experiment.mode == "augmented":
                wandb_logger = WandbLogger(project="test_mask", name=f"augmentation_{cfg.experiment.noise_percentage*100}%_model_{i}")
            else:
                wandb_logger = WandbLogger(project="test_mask", name=f"{model_name}_{dataset_name}_target_sparsity_{target_sparsity}_model_{i}")

            trainer = Trainer(logger=wandb_logger)
            
            # test original model only once
            #if counter == 1:
            print("\nOriginal Model:")
            classifier_network.load_state_dict(checkpoint)
            classifier_network.eval()
            original_metrics = test_classifier(classifier_network, test, n_classes, batch_size, device, remapping)

            print(f"\nReconstructing model {i}")
            current_acc_ret, current_compression_ratio, current_orig_acc, current_inj_acc, current_sparsity = generate_test_mask(cfg, checkpoint, tokenizer, trainer, sane_model, classifier_network, test, n_classes, batch_size, device, remapping, i, original_metrics, target_sparsity)
            accuracy_retentions.append(current_acc_ret)
            compression_rates.append(current_compression_ratio)
            original_accuracies.append(current_orig_acc)
            injected_accuracies.append(current_inj_acc)
            sparsities.append(current_sparsity)

        print(f"\nAverage original accuracy: {(sum(original_accuracies) / len(original_accuracies))*100:.2f}%")
        print(f"\nAverage injected accuracy: {(sum(injected_accuracies) / len(injected_accuracies))*100:.2f}%")
        print(f"\nAverage accuracy retention: {(sum(accuracy_retentions) / len(accuracy_retentions))*100:.2f}%")
        print(f"\nAverage compression rate: {(sum(compression_rates) / len(compression_rates)):.2f}x")
        print(f"\nAverage sparsity: {(sum(sparsities) / len(sparsities)):.2f}%")



if __name__ == "__main__":
    main()