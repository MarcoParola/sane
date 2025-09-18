import torch
import math
import lightning as L
from src.models.transformers.gpt2 import GPTransformer
from src.losses.loss import MaskedReconLoss
from src.models.sane.positional_embs import PositionalEmbs
from src.models.sane.projection_head import ProjectionHead


class Sane(L.LightningModule):
    def __init__(self, conf: dict = None,
                 idim: int = 288, edim: int = 2048, n_head: int = 16, n_blocks: int = 8,
                 latdim: int = 128, wsize: int = 256, dropout: float = 0.0,
                 max_positions: list[int] = [55000, 100, 550]
                 ):
        super().__init__()
        self.save_hyperparameters()
        self.conf = conf

        # model
        self.tokenizer = torch.nn.Linear(idim, edim)
        self.transformer_encoder = GPTransformer(n_blocks, wsize, n_head, edim, dropout, bias=False, causal=False)
        self.encoder_comp = torch.nn.Linear(edim, latdim)
        self.decoder_comp = torch.nn.Linear(latdim, edim)
        self.transfomer_decoder = GPTransformer(n_blocks, wsize, n_head, edim, dropout, bias=False, causal=False)
        self.detokenizer = torch.nn.Linear(edim, idim)
        self.pe = PositionalEmbs(max_positions, edim)
        self.projection_head = ProjectionHead(latdim, wsize, odim=30)
        self.dropout = torch.nn.Dropout(dropout)

        # loss
        self.criterion = MaskedReconLoss(conf.training.loss_reduction)
            
        # taken from Kaparthy's GPT2 implementation:
        self.apply(self._init_weights)
        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith("projection.weight"):
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * n_blocks))

        # test-time collectors
        self._test_reconwindows: list[torch.Tensor] = []
        self._test_positions: list[torch.Tensor] = []
        self._test_embeddings: list[torch.Tensor] = []

    
    def _init_weights(self, module):
        if isinstance(module, torch.nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, torch.nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def encode(self, x: torch.Tensor, p: torch.Tensor, m: torch.Tensor = None):
        # get a token from x
        x = self.tokenizer(x)
        # add positional encoding
        x = self.pe(x, p)
        x = self.dropout(x)
        x = self.transformer_encoder(x,m)
        x = self.encoder_comp(x)
        # return compressed encoding of token+posemb
        return x
    
    def decode(self, z: torch.Tensor, p: torch.Tensor, m: torch.Tensor = None):
        # decode compressed encoding of token+posemb
        z = self.decoder_comp(z)
        # add positional encoding
        z = self.pe(z, p)
        z = self.dropout(z)
        z = self.transfomer_decoder(z,m)
        z = self.detokenizer(z)
        # return detokenized decoded weights
        return z
    
    def forward(self, x: torch.Tensor, p: torch.Tensor, m: torch.Tensor = None):
        z = self.encode(x, p, m)
        zp = self.projection_head(z)
        y = self.decode(z, p, m)
        return z,y,zp
    
    def forward_embeddings(self, x: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
        x = self.encode(x,p)
        return x.mean(dim=1)
    
    def _common_step(self, batch, batch_idx, stage):
        t, m, p = batch
        z, y, zp = self(t, p, m=None)
        loss = self.criterion(y, t, m)
        self.log(f"{stage}_loss", loss, on_epoch=True, prog_bar=True)
        return loss 

    def training_step(self, batch, batch_idx):
        loss = self._common_step(batch, batch_idx, "train")
        #self.log("val_loss", loss, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        return self._common_step(batch, batch_idx, "val")
    
    def on_test_start(self):
        self._test_reconwindows.clear()
        self._test_positions.clear()
        self._test_embeddings.clear()

    def test_step(self, batch, batch_idx):
        self._common_step(batch, batch_idx, "test")
        t, m, p = batch
        z, y, zp = self(t, p, m=None)
        self._test_reconwindows.append(y.detach().cpu().reshape(y.shape[0]*y.shape[1], -1))
        self._test_positions.append(p.detach().cpu().reshape(p.shape[0]*p.shape[1], -1))
        self._test_embeddings.append(z.detach().cpu().reshape(z.shape[0]*z.shape[1], -1))

    def configure_optimizers(self):
        trainable_params = {pn: p for pn,p in self.named_parameters() if p.requires_grad}
        decay_params = [p for _,p in trainable_params.items() if p.dim() >= 2]
        nodec_params = [p for _,p in trainable_params.items() if p.dim() < 2]
        optim_groups = [
            {"params": decay_params, "weight_decay": self.conf.optimizer.wd},
            {"params": nodec_params, "weight_decay": 0.0}
        ]
        total_steps = self.trainer.estimated_stepping_batches
        optimizer = torch.optim.AdamW(params=optim_groups, lr=self.conf.optimizer.lr)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, self.conf.optimizer.lr, total_steps)
        
        lr_scheduler_config = {
            "scheduler": scheduler,
            "interval": "step",
            "frequency": 1
        }

        return [optimizer], [lr_scheduler_config]
    

    def get_test_outputs(self):
        """
        Returns:
            reconwindows: (N, idim)
            positions:    (N, position_dims)
            embeddings:   (N, latdim)
        """
        recon = torch.cat(self._test_reconwindows, dim=0) if self._test_reconwindows else torch.empty(0)
        pos = torch.cat(self._test_positions, dim=0) if self._test_positions else torch.empty(0)
        emb = torch.cat(self._test_embeddings, dim=0) if self._test_embeddings else torch.empty(0)
        return recon, pos, emb   