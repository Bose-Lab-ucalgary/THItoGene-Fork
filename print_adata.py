import anndata as ad
import os

path = "../../data/HERST_preprocess/3CA_genes/train"
file = "MEND145_preprocessed.h5ad"
adata = ad.read_h5ad(os.path.join(path, file))
print(adata.n_obs, adata.n_vars)
# print(adata)


# tas_path = "../../../../tahsin/HEST/data/hest_human_only_jul11/patched_adata"
# og_adata = ad.read_h5ad(os.path.join(tas_path, file))

# print(og_adata)