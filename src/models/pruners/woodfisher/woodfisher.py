import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, SubsetRandomSampler
import torch.optim as optim
import math
from math import ceil

class WoodburryFisherPruner:
    def __init__(self, model, 
                 fisher_subsample_size=1000, 
                 fisher_mini_bsz=1, 
                 fisher_damp=1e-5,
                 aux_gpu_id=-1,
                 weight_only=True):

        self._model = model
        self._fisher_subsample_size = fisher_subsample_size
        self._fisher_mini_bsz = fisher_mini_bsz
        self._fisher_damp = fisher_damp
        self._aux_gpu_id = aux_gpu_id
        self._weight_only = weight_only

        self._fisher_inv_diag = None


    def _release_grads(self):
        optim.SGD(self._model.parameters(), lr=1e-10).zero_grad()


    def flatten_tensor_list(self, tensors):
        flattened = []
        for tensor in tensors:
            flattened.append(tensor.view(-1))
        return torch.cat(flattened, 0)


    def _compute_sample_fisher(self, loss, return_outer_product=False):
        params = [p for m in self._modules for n, p in m.named_parameters()
                  if not (self._weight_only and 'bias' in n)]

        grads = torch.autograd.grad(loss, params)
        for idx, module in enumerate(self._modules):
            grads[idx].data.mul_(module.weight_mask)
        grads = self.flatten_tensor_list(grads)

        self._num_params = len(grads)

        if return_outer_product:
            return torch.ger(grads, grads)
        return grads


    def _get_param_stat(self, param, param_mask, fisher_inv_diag, param_idx):
        inv_fisher_diag_entry = fisher_inv_diag[param_idx:param_idx + param.numel()].view_as(param).to(param.device)
        return ((param ** 2) / (inv_fisher_diag_entry + 1e-10) + 1e-10) * param_mask

    def _get_pruned_wts_scaled_basis(self, pruned_params, flattened_params):
        return -1 * torch.div(torch.mul(pruned_params, flattened_params), self._fisher_inv_diag)

    def _add_outer_products_efficient_v1(self, mat, vec, num_parts=2):
        piece = math.ceil(len(vec) / num_parts)
        for i in range(num_parts):
            for j in range(num_parts):
                mat[i*piece:min((i+1)*piece, len(vec)),
                    j*piece:min((j+1)*piece, len(vec))].add_(
                    torch.ger(vec[i*piece:min((i+1)*piece, len(vec))],
                              vec[j*piece:min((j+1)*piece, len(vec))])
                )

    def _compute_woodburry_fisher_inverse(self, dset, subset_inds, device, num_workers=0):
        self._model.to(device)
        goal = self._fisher_subsample_size

        loader = DataLoader(dset, batch_size=self._fisher_mini_bsz, num_workers=num_workers,
                            sampler=SubsetRandomSampler(subset_inds))

        criterion = F.nll_loss
        aux_device = torch.device(f"cuda:{self._aux_gpu_id}") if self._aux_gpu_id != -1 else torch.device('cpu')

        self._fisher_inv = None
        num_batches = 0
        num_samples = 0

        for x, y in loader:
            self._release_grads()
            x, y = x.to(device), y.to(device)
            output = self._model(x)
            loss = criterion(output, y)
            sample_grads = self._compute_sample_fisher(loss)

            if aux_device != torch.device('cpu'):
                sample_grads = sample_grads.to(aux_device)

            if num_batches == 0:
                norm = self._fisher_damp ** 2
                self._fisher_inv = torch.ger(sample_grads, sample_grads).mul_(1/norm).div_(
                    goal + (sample_grads.dot(sample_grads)/self._fisher_damp)
                )
                self._fisher_inv.diagonal().sub_(1/self._fisher_damp)
                self._fisher_inv.mul_(-1)
            else:
                cache = torch.matmul(self._fisher_inv, sample_grads)
                cache.div_((goal + sample_grads.dot(cache))**0.5)
                self._fisher_inv.sub_(torch.ger(cache, cache))
                del cache

            num_batches += 1
            num_samples += self._fisher_mini_bsz
            if num_samples == goal * self._fisher_mini_bsz:
                break

        self._fisher_inv_diag = self._fisher_inv.diagonal()

    def percentile(self, tensor, p):
        """
        Returns percentile of tensor elements

        Arguments:
            tensor {torch.Tensor} -- a tensor to compute percentile
            p {float} -- percentile (values in [0,1])
        """
        if p > 1.:
            raise ValueError(f'Percentile parameter p expected to be in [0, 1], found {p:.5f}')
        k = ceil(tensor.numel() * (1 - p))
        if p == 0:
            return -1  # by convention all param_stats >= 0
        # topk returns a tuple: (values, indices) and essentially amongst the k percentile we are returning the smallest value
        return torch.topk(tensor.view(-1), k)[0][-1]

    def _get_pruning_mask(self, param_stats, sparsity):
        if param_stats is None: return None
        threshold = self.percentile(param_stats, sparsity)
        return (param_stats > threshold).float()

    def prune(self, dset, subset_inds, sparsity_level, device):
        # unwrap prunable modules
        self._modules = [module for module in self._model.modules() if hasattr(module, 'weight_mask')]

        # ensure that the model is not in training mode, this is importance, because
        # otherwise the pruning procedure will interfere and affect the batch-norm statistics
        assert not self._model.training

        #############################################################
        # Step 1. Computer full fisher inverse via woodburry
        self._compute_woodburry_fisher_inverse(dset, subset_inds, device)

        assert self._num_params == self._fisher_inv_diag.shape[0]
        self._param_idx = 0

        flat_pruned_weights_list = []
        flat_module_weights_list = []
        module_shapes_list = []
        module_param_indices_list = []
        prune_masks = []
        past_weight_masks = []

        #############################################################

        # Step 2. Compute param stats and masks at once.

        for idx, module in enumerate(self._modules):
            # print(f'module is {module}')
            level = sparsity_level

            # multiplying by the current mask makes the corresponding statistic
            # of those weights zero and keeps them removed.
            past_weight_masks.append(module.weight_mask)
            module_param_indices_list.append(self._param_idx)
            assert self._weight_only
            module_shapes_list.append(module.weight.shape)

            w_stat = self._get_param_stat(module.weight, module.weight_mask, self._fisher_inv_diag, self._param_idx)
            self._param_idx += module.weight.numel()

            module.weight_mask = self._get_pruning_mask(w_stat, level)


        #############################################################
        # Step 3. Now that sparsification masks have been computed,
        # put them together in a list, and apply the requisite OBS update to other remaining weights

        for idx, module in enumerate(self._modules):
            assert self._weight_only
            pruned_weights = past_weight_masks[idx] - module.weight_mask
            prune_mask = past_weight_masks[idx] > module.weight_mask
            prune_masks.append(prune_mask)
            # print(f'pruned_weights are {pruned_weights}')
            pruned_weights = pruned_weights.flatten().float()
            flat_pruned_weights_list.append(pruned_weights)
            flat_module_weights_list.append(module.weight.flatten())

        module_param_indices_list.append(self._param_idx)

        flat_pruned_weights_list = self.flatten_tensor_list(flat_pruned_weights_list)
        flat_module_weights_list = self.flatten_tensor_list(flat_module_weights_list)

        # compute the weight update across all modules
        scaled_basis_vector = self._get_pruned_wts_scaled_basis(flat_pruned_weights_list, flat_module_weights_list)
        weight_updates = self._fisher_inv @ scaled_basis_vector

        # now apply the respective module wise weight update
        for idx, module in enumerate(self._modules):
            weight_update = weight_updates[module_param_indices_list[idx]:module_param_indices_list[idx + 1]]
            weight_update = weight_update.view_as(module.weight)
            weight_update[prune_masks[idx]] = (-1 * module.weight.data[prune_masks[idx]])

            # print('weight before is ', module.weight)
            with torch.no_grad():
                module.weight[:] = module.weight.data + weight_update

        self._release_grads()

        # check if all the params whose fisher inverse was computed their value has been taken
        assert self._param_idx == self._fisher_inv_diag.shape[0]

        del self._fisher_inv