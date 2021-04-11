import pandas as pd
import numpy as np
from anndata import AnnData
import scvelo as scv
import scanpy as sc
import matplotlib.pyplot as plt

N_GENES = 500
N_SAMPLES = 500

adata = AnnData(
     pd.DataFrame(
         np.random.randint(0, 100, (N_SAMPLES, N_GENES)),
         columns=[f"gene_{id}" for id in range(N_GENES)],
         index=[f"cell_{id}" for id in range(N_GENES)]
     ),
     layers={
         "spliced": np.random.randint(0, 100, (N_SAMPLES, N_GENES)),
         "unspliced": np.random.randint(0, 100, (N_SAMPLES, N_GENES)),
     }
)



# define "velocity_genes"
adata.var["velocity_genes"] = np.random.choice([True, False], size=N_GENES)
#print(f"Velocity genes: {adata.var['velocity_genes'].head()}")
scv.pp.normalize_per_cell(adata, enforce = True)
scv.pp.log1p(adata)
scv.pp.moments(adata, n_neighbors = 5)
scv.tl.recover_dynamics(adata, var_names="velocity_genes")
#adata.var.head()
