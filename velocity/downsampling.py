import pandas as pd
import numpy as np
from anndata import AnnData
import scvelo as scv
import scanpy as sc
import matplotlib.pyplot as plt

#downsample ada1 to ada2 by each cluster

adata = sc.read_h5ad('./adata_filtered.h5ad')
ada1 = adata[adata.obs.groupID == '1']
ada2 = adata[adata.obs.groupID == '2']

celltypes = np.unique(np.asarray(ada1.obs.celltype)).tolist()
exp_cell = [ct for ct in celltypes if len(ada1[ada1.obs.celltype == ct]) < len(ada2[ada2.obs.celltype==ct])]
celltype.remove(exp_cell)

down_list = []
for ct in celltypes:
    if len(ada1[ada1.obs.celltype == ct]) < len(ada2[ada2.obs.celltype==ct]):
        down1 = np.asarray(ada1.obs_names[ada1.obs.celltype==ct])
    else:
        down1 = np.random.choice(ada1.obs_names[ada1.obs.celltype==ct], len(ada2[ada2.obs.celltype==ct]), replace=False)
    down_list.extend(down1)
    
ada1 = ada1[down_list, :]
