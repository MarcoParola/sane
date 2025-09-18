import torch

class NT_Xent(torch.nn.Module):
    def __init__(self, temperature: float, positive_only: bool = False):
        super(NT_Xent, self).__init__()
        self.temperature = temperature; self.positive_only = positive_only
        self.ce = torch.nn.CrossEntropyLoss(reduction="sum")
        self.similarity = torch.nn.functional.cosine_similarity
    
    def mask_correlated_samples(self, batchsize: int):
        N = 2*batchsize
        mask: torch.Tensor = torch.ones((N, N), dtype=bool) # boolean indexing matrix for negative samples
        # 1 samples are the one considered as negative, so the one the model should diverge from
        mask = mask.fill_diagonal_(0)
        for i in range(batchsize):
            mask[i, batchsize + i] = 0 # diagonal batch_size (upper)
            mask[batchsize + i, i] = 0 # diagonal -batch_size (lower)
        return mask
    
    def forward(self, zi: torch.Tensor, zj: torch.Tensor):
        '''zi and zj are two embeddings of the same model in different views (after applying augmentation), shape: (Bs, Cn)'''
        batch_size = zi.shape[0]
        mask = self.mask_correlated_samples(batch_size)
        N = 2*batch_size
        z = torch.cat([zi, zj], dim=0) # along batch dimension
        
        # results in [Bs, Bs] where each element ij is the similarity of 
        # zi@i with zj@i(first batch-size elements) 
        # and zj@i with zi@j(second batch-size elements)
        sim: torch.Tensor = self.similarity(z[None, :, :], z[:, None, :], dim=-1) / self.temperature
        sim_i_j = sim.diag(batch_size)
        sim_j_i = sim.diag(-batch_size)

        positive = torch.cat([sim_i_j, sim_j_i], dim=0).reshape(N, 1)
        negative = sim[mask].reshape(N, -1)

        labels = torch.zeros(N).to(positive.device).long()
        if self.positive_only: labels = labels.unsqueeze(dim=1)

        logits = torch.cat([positive, negative], dim=1) # positive samples always at position 0 to match labels
        loss = self.ce(positive, labels) if self.positive_only else self.ce(logits, labels)
        loss /= N
        return loss

class MVNT_Xent(torch.nn.Module):
    def __init__(self, temperature: float):
        super(MVNT_Xent, self).__init__()
        self.temperature = temperature
        self.similarity = torch.nn.functional.cosine_similarity
    
    def get_masks(self, batchsize: int, nviews: int, device: str):
        '''
            * Computes N as nviews * batchsize
            * generates the NxN boolean selector mask of positive samples including self-similarities setting them to True
            * negates to obtain the negative samples that also excludes self-similarities which need to be ignored in both positive
            and negative
        '''
        N = nviews * batchsize
        mask: torch.Tensor = torch.full((N,N), False, device=device, dtype=bool)
        positive_indexes = torch.arange(N, device=device) % batchsize
        mask = (positive_indexes[None, :] == positive_indexes[:, None])
        return ~mask, mask & ~torch.eye(N, dtype=bool, device=device)
    
    def forward(self, *zviews: torch.Tensor):
        '''computes NT_Xent between n views of the original'''
        nviews = len(zviews)
        assert nviews > 1, f"Need at least 2 views of the feature, received {nviews}"
        batchsize = zviews[0].shape[0]
        z = torch.cat(zviews, dim=0)
        N = z.shape[0] # equal to (nrandom * batchsize) + batchsize (original view) - anchors
        nmask, pmask = self.get_masks(batchsize, nviews, z.device)
        
        sim: torch.Tensor = self.similarity(z[None, :, :], z[:, None, :], dim=-1) / self.temperature

        positive = sim[pmask].reshape(N, nviews-1) # 1 score for each of the nviews -1 for the self-similarity times N anchors
        negative = sim[nmask].reshape(N, -1) # N-nviews scores for each of the N anchors will consequently be the negative scores

        logits = torch.cat([positive, negative], dim=1)
        # this time there will be multiple positive elements (the first nviews-1 in each of the N anchors)
        logprobs = torch.nn.functional.log_softmax(logits, dim=1) # exp(sim_ij)/sum(exp(sim_ik)) for all k
        elementwise_loss = -logprobs[:, :nviews-1].sum(dim=1) # maximize summation of positive samples to get as close to 1 similarities

        return elementwise_loss.mean()

class MaskedReconLoss(torch.nn.Module):
    def __init__(self, reduction: str):
        super(MaskedReconLoss, self).__init__()
        self.mse = torch.nn.MSELoss(reduction=reduction)
        self.loss_mean = None
    
    def forward(self, output: torch.Tensor, target: torch.Tensor, mask: torch.Tensor):
        assert (
            output.shape == target.shape == mask.shape
        ), f"MSE loss error: prediction and target don't have the same shape. output {output.shape} vs target {target.shape} vs mask {mask.shape}"

        loss = self.mse(mask*output, target)
        # rsq part with torchmetrics.functional.explained_variance non sembra utilizzato
        return loss
    
    # def set_mean_loss(self, data: torch.Tensor, mask: torch.Tensor):
    #     """
    #     #TODO - l'implementazione sembra incompleta
    #     """
    #     # check that data are tensor..
    #     assert isinstance(data, torch.Tensor)
    #     w_mean = data.mean(dim=0)  # compute over samples (dim0)
    #     # scale up to same size as data
    #     data_mean = repeat(w_mean, "l d -> n l d", n=data.shape[0])
    #     out_mean = self.forward(data_mean, data, mask)

    #     # compute mean
    #     print(f" mean loss: {out_mean['loss_recon']}")

    #     self.loss_mean = out_mean["loss_recon"]


class GammaContrastReconLoss(torch.nn.Module):
    def __init__(
        self, gamma: float, reduction: str, temperature: float, contrast: str = "simclr", 
        z_var_penalty: float = 0.0, z_norm_penalty: float = 0.0
    ):
        '''
        :param conrast: can be one of "simclr" which will use standard NT_Xent; "positive" which will use positive_only pulling of
        positive samples in NT_Xent; "mvsimclr" which will use multi-view contrastive loss
        :param reduction: reduction method to be used for Mask-relevant MSE
        :param temperature: temperature to be used in NT_Xent or MVNT_Xent
        '''
        super(GammaContrastReconLoss, self).__init__()
        assert 0 <= gamma <= 1
        self.gamma = gamma; self.zvp = z_var_penalty; self.znp = z_norm_penalty
        self.contrast_loss = MVNT_Xent(temperature) if contrast == "mvsimclr" else NT_Xent(temperature, positive_only=(contrast == "positive"))
        self.recon_loss = MaskedReconLoss(reduction)
    
    def forward(self, views: list[torch.Tensor], y: torch.Tensor, t: torch.Tensor, m: torch.Tensor):
        if isinstance(self.contrast_loss, NT_Xent): assert len(views) == 2, f"2 views are required for selected contrastive loss"
        reconloss = self.recon_loss(y, t, m) # reconstruction loss using MSE between y reconstructed and target (original)
        contrloss = self.contrast_loss(*views) # constrastive loss to guide convergence for positive samples and divergence for negative ones
        totalloss = self.gamma * contrloss + (1-self.gamma)*reconloss

        znorm = torch.linalg.norm(views[0].view(views[0].shape[0], -1), ord=2, dim=1).mean() # original view penalty based on norm
        zvar = torch.mean(torch.var(views[0].view(views[0].shape[0], -1), dim=0)) # original view penalty based on variance

        totalloss = totalloss + self.zvp * zvar + self.znp * znorm
        return totalloss
    
    # def set_mean_loss(self, weights: torch.Tensor, mask=None) -> None:
    #     """
    #     Helper function to set mean loss in reconstruction loss - Sembra incompleta
    #     """
    #     # if mask not set, set it to all ones
    #     if mask is None:
    #         mask = torch.ones(weights.shape)
    #     # call mean_loss function
    #     self.loss_recon.set_mean_loss(weights, mask=mask)