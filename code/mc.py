# salloc --mem=32G --time=00:10:00 --gres=gpu:1
# TODO - try averaging prod_weights and comp_weights across subjects before computing cdist
# TODO - i'm assuming pdist using triu with offset 1 as ward expects
# TODO - not sure whether to set nans to 0 or 1e-8. prob should check where nans are occuring anyways

# # salloc --mem=64G --time=00:15:00
# import numpy as np
# from scipy.cluster.hierarchy import ward
# modelname = 'model-gpt2-medium_layer-0.0'
# D = np.load(f'dists_{modelname}_mode-1.npy')
# Z = ward(D)
# print(Z.shape)
# np.save(f'linkage_{modelname}.npy', Z)

import resource

import h5py
import numpy as np
import torch
from constants import SUBS_STRANGERS
from tqdm import tqdm
from util.path import Path

# NOTE this is prod-comp on self.

modelname = "model-gpt2-medium_layer-0.75"
modelname = "model-gpt2-medium_layer-0.0"

respath = Path(
    root="encoding",
    sub="000",
    datatype=modelname,
    ext=".hdf5",
)


def print_cuda_mem(preamble=None):
    # torch.cuda.memory_summary()
    denom = 1024 * 1024 * 1024
    if preamble is not None:
        print(preamble)
    print("memory_allocated: %fGB" % (torch.cuda.memory_allocated(0) / denom))
    print("memory_reserved: %fGB" % (torch.cuda.memory_reserved(0) / denom))
    print("max_memory_reserved: %fGB" % (torch.cuda.max_memory_reserved(0) / denom))
    print(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / denom)
    print()


print_cuda_mem("start")

# mode 1
dists = torch.zeros((81924, 81924), dtype=torch.float32, device="cuda")

# # mode 2
# prod_weights = torch.zeros((1024, 81924), dtype=torch.float32)
# comp_weights = torch.zeros((1024, 81924), dtype=torch.float32)

print_cuda_mem("alloc dists")

for sub in tqdm(SUBS_STRANGERS):
    respath.update(sub=f"{sub:03d}")
    with h5py.File(respath, "r") as f:
        weights_prod = f["cv_weights_prod"][...].mean(0)
        weights_comp = f["cv_weights_comp"][...].mean(0)
        corr_prod = f["cv_scores_prod"][...].mean(0)
        corr_comp = f["cv_scores_comp"][...].mean(0)

    norms = np.linalg.norm(weights_prod, ord=2, axis=0)
    weights_prod /= norms
    weights_prod = np.nan_to_num(weights_prod, copy=False, nan=0)
    r2_prod = np.square(corr_prod)
    r2_prod = np.clip(r2_prod, a_min=0, a_max=None, out=r2_prod)
    weights_prod *= r2_prod

    norms = np.linalg.norm(weights_comp, ord=2, axis=0)
    weights_comp /= norms
    weights_comp = np.nan_to_num(weights_comp, copy=False, nan=0)
    r2_comp = np.square(corr_comp)
    r2_comp = np.clip(r2_comp, a_min=0, a_max=None, out=r2_comp)
    weights_comp *= r2_comp

    # mode 1
    wp = torch.tensor(weights_prod.T, device="cuda")
    wc = torch.tensor(weights_comp.T, device="cuda")
    dist = torch.cdist(wp, wc, p=2)
    dists += dist

    # # mode 2
    # prod_weights += weights_prod
    # comp_weights += weights_comp

n = len(SUBS_STRANGERS)

# mode 1
dists /= n  # get mean
del wp, wc, dist
torch.cuda.empty_cache()

# # mode 2
# prod_weights /= n
# comp_weights /= n
# wp = torch.tensor(prod_weights.T, device="cuda")
# wc = torch.tensor(comp_weights.T, device="cuda")
# dists = torch.cdist(wp, wc, p=2)

torch.save(dists, f"dists_{modelname}_mode-1.pt")
# dists = torch.load("mc_dist.pt").to("cuda")

print_cuda_mem("filled dists")

start = 0
tri_ids = torch.triu_indices(81924, 81924, offset=1, dtype=torch.int32, device="cuda")
dists_flat = np.zeros(tri_ids.shape[1], dtype=np.float32)
print_cuda_mem("before")
for chunk in tri_ids.chunk(6, dim=-1):
    part = dists[chunk[0], chunk[1]].numpy(force=True)
    end = chunk.shape[1]
    dists_flat[start : start + end] = part
    print(start, start + end)
    start += end
print_cuda_mem("after")

np.save(f"dists_{modelname}_mode-1.npy", dists_flat)
