from contextlib import nullcontext
import numpy as np
import torch
from src.models.pruners.chita.utils import *
from src.models.pruners.chita.heuristic_lsblock import *

class CHITA:

    def __init__(self,model,params,prun_dataloader,device,
                ngrads=500,
                criterion=torch.nn.functional.nll_loss,
                blocksize=500,
                lambda2=0.0001):
        self.model = model
        self.params = params 
        self.prun_dataloader = prun_dataloader
        self.criterion = criterion
        self.ngrads = ngrads
        self.blocksize = blocksize
        self.lambda2 = lambda2*ngrads/2
        self.device = device 
        self.grads = None

        self.block_list = get_blocklist(self.model,self.params,self.blocksize)

        
    def prune(self,mask,sparsity,grads=None):
        original_weight = get_pvec(self.model, self.params)
        if mask is None:
            mask = torch.ones_like(original_weight).cpu() != 0
        w1 = original_weight.to('cpu').numpy()
        d = len(w1)
        k = int((1-sparsity)*original_weight.numel())

        zero_grads(self.model)
        self.model.eval()

        
        if grads is None and self.grads is None:
            with self.model.no_sync() if isinstance(self.model,torch.nn.parallel.DistributedDataParallel) else nullcontext() as gs:
                grads = torch.zeros((self.ngrads, d), device='cpu')
                for i, batch in enumerate(self.prun_dataloader):
                    x, y = batch
                    x = x.to(self.device)
                    y = y.to(self.device)
                    loss = self.criterion(self.model(x), y)
                    loss.backward()
                    grads[i] = get_gvec(self.model, self.params).to('cpu')
                    zero_grads(self.model)

                    if (i + 1) % self.ngrads == 0:
                        break
                
            grads = grads.numpy()
        self.grads = grads
        w1 = w1.astype(self.grads.dtype)
        
        y=grads@w1
        beta_tilde2=np.copy(w1)
        beta_tilde1 = np.zeros_like(w1)

        
        alpha_vec = np.zeros_like(w1)

        w_pruned, obj, _, sol_time = Heuristic_LSBlock(w1,grads,w1,k,alpha=alpha_vec,lambda1=0,lambda2=self.lambda2,M=np.inf, beta_tilde1=beta_tilde1, 
                        beta_tilde2=beta_tilde2, use_prune=True,block_list=self.block_list, split_type=1)

        new_mask = torch.from_numpy(w_pruned != 0)

        return w_pruned,new_mask