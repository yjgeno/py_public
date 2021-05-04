import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams
import scvelo as scv
import scanpy as sc

sc.settings.verbosity = 3  # verbosity: errors (0), warnings (1), info (2), hints (3)
sc.settings.set_figure_params(dpi=200, frameon=False, figsize=(15, 12), facecolor='white') 

ada1 = sc.read_h5ad('./ada1_raw.h5ad')
ada1.X = ada1.X.astype('float64')
ada2 = sc.read_h5ad('./ada2_raw.h5ad')
ada2.X = ada2.X.astype('float64')

i = 1
for ada in [ada1, ada2]:
        scv.pp.normalize_per_cell(ada, enforce = True)
	ada.obsm['X_tsne'] = ada.obsm['X_tsne'][:, :2]
	sc.tl.pca(ada, svd_solver='arpack')
	sc.pp.neighbors(ada, n_neighbors=4, n_pcs=20)
	sc.tl.draw_graph(ada, layout='fa', random_state=1) # add fa embedding
	sc.tl.paga(ada, groups='celltype')
	
	sc.pl.paga_compare(
    ada2, color='celltype', basis='X_draw_graph_fa', threshold=0.03, title='', right_margin=0.2, size=10, edge_width_scale=0.5,
    legend_fontsize=12, fontsize=12, frameon=False, edges=True, save='ada{}_fa.png'.format(i))
	
  #default tsne embedding
	sc.pl.paga_compare(
    ada2, color='celltype', basis='X_tsne', threshold=0.03, title='', right_margin=0.2, size=10, edge_width_scale=0.5,
    legend_fontsize=12, fontsize=12, frameon=False, edges=True, save='ada{}_tsne.png'.format(i))
	i += 1
  
  
  
