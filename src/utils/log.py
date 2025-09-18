from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.loggers import TensorBoardLogger

def get_loggers(cfg, run_group=None, run_name=None):
    """Returns a list of loggers
    cfg: hydra config
    """
    loggers = []

    if cfg.log.wandb:
        wandb_logger = WandbLogger(
            project = cfg.wandb.project,
            name = run_name,
            group = run_group
        )
        loggers.append(wandb_logger)

    if cfg.log.tensorboard:
        tensorboard_logger = TensorBoardLogger(cfg.log.dir, name=cfg.wandb.project)
        loggers.append(tensorboard_logger)

    return loggers