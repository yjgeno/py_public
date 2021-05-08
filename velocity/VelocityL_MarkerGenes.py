import pandas as pd
import numpy as np
import anndata
import scvelo as scv
import scanpy as sc
import matplotlib
import matplotlib.pyplot as plt
from scipy.stats import norm
import math

ada1 = sc.read_h5ad('./ada1_built_downs.h5ad')
#ada1.var_names = ada1.var.gene_symbols
ada2 = sc.read_h5ad('./ada2_built_downs.h5ad')
#ada2.var_names = ada2.var.gene_symbols

def velogene_ada(ada): 
    V = ada.layers['velocity']
    tmp_filter = np.invert(np.isnan(np.sum(V, axis=0))) # filter NaN
    
    tmp_filter &= np.array(ada.var['velocity_genes'], dtype=bool)
    return ada[:, tmp_filter]

ada1_v = velogene_ada(ada1) #velocity genes
ada2_v = velogene_ada(ada2)
ada1_genes = np.char.upper(np.array(ada1_v.var_names).tolist()).tolist()
ada2_genes = np.char.upper(np.array(ada2_v.var_names).tolist()).tolist()
#len(ada1_genes), len(ada2_genes)
ada_genes = list(set(ada1_genes) & set(ada2_genes))
#len(ada_genes)
ada1_v.var_names = np.char.upper(np.array(ada1_v.var_names).tolist()) #upper case
ada2_v.var_names = np.char.upper(np.array(ada2_v.var_names).tolist())

#velocity genes in same length
ada1 = ada1_v[:, ada1_v.var_names.isin (ada_genes)]
ada2 = ada2_v[:, ada2_v.var_names.isin (ada_genes)]

def v_length(ada): # velocity_length
    V = ada.layers['velocity']
    print ('V_shape: ', V.shape)
    tmp_filter = np.invert(np.isnan(np.sum(V, axis=0))) # filter NaN
    print ('V_shape_filtered_1: ', V[:, tmp_filter].shape)
    
    tmp_filter &= np.array(ada.var['velocity_genes'], dtype=bool)
    V = V[:, tmp_filter]
    print ('V_shape_filtered_2: ', V.shape)
    
    V -= V.mean(1)[:, None] # - row means (centered)
    V_norm = np.linalg.norm(V, axis=1)
    #print ('max(V_norm): ', max(V_norm))
    return V_norm.round(2)
  
for ada in [ada1, ada2]:
    scv.tl.velocity_confidence(ada)
    normalize = matplotlib.colors.Normalize(vmin=0, vmax=40)
    scv.pl.scatter(ada, c='velocity_length', cmap='seismic', 
                   norm=normalize, perc=[2, 98], dpi=200, save='ada_SAMElength.png')
    df = pd.DataFrame(index = np.array(ada.obs.celltype), data = v_length(ada))
    df.to_csv('./ada_SAMElength.csv', header = False)

#genes of interest
genelist1 = ['Rprl3', 'Gm4989', 'Gm6472', 'Pxdn', 'Ccl25', 'Gm9625', 'Cpm', 
             'Hsd17b13', 'Mptx1', 'Acp1', 'Prkd1', 'Gm8730', 'Rplp0-ps1', 'Mtus2', 'Sycn', 'Lrg1', 'Nnt', 'Reg3g', 'la2g5']
genelist2 = ['Socs3', 'Foxm1', 'Plk1', 'Reg3g', 'Reg3b', 'Reg4', 'Cdc25b', 'Hpgd', 'Ugt2b5', 'Fads1']
genelist3 = ['FOXM1', 'TGFB1', 'IL10RA', 'IL10RB', 'IL22RA1', 'IL22RA2', 'ODC1', 'MMP1', 'ABCG2', 'RNF43']
genelist = []
for l in [np.char.upper(genelist1), np.char.upper(genelist2), genelist3]:
    genelist.extend(l)
ada_genes_for_test = np.char.capitalize(list(set(ada_genes) & set(genelist))).tolist()
ada_genes_for_test = np.char.upper(ada_genes_for_test) #interested genes also in velocity genes

ada1 = ada1_v[:, ada1_v.var_names.isin (ada_genes_for_test)]
ada2 = ada2_v[:, ada2_v.var_names.isin (ada_genes_for_test)]

kwargs = dict(linewidth=1.5, color_map='coolwarm', figsize= (6, 5), colorbar=False, size=64, 
               dpi=200, legend_fontsize=12, fontsize=14, smooth=30)

for g in ada_genes_for_test:
    #xmax = max(ada2[:, ada2.var_names.isin ([g])].layers['spliced'].tolist())[0]
    #ymax = max(ada2[:, ada2.var_names.isin ([g])].layers['unspliced'].tolist())[0]
    for ada in [ada1, ada2]:
        scv.pl.scatter(ada, g, color='velocity', **kwargs,
                       xlim = (0, math.log10(xmax)), ylim = (0, math.log10(ymax)), save='{}_{}_dynamics.png'.format(g, namestr(ada, globals())[0]))
        
        
        

