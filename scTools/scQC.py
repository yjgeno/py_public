import numpy as np
import matplotlib.pyplot as plt
import scanpy as sc
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import summary_table

def scQC(adata, mtThreshold = 0.1, minLSize = 1000, a = 0.05, plot = False):
    
    def plotQC(X, y, y_fit, y_fit_low, y_fit_upp):  
        fig = plt.figure(figsize=(3,3))
        plt.plot(X, y, 'o', markersize=2)
        plt.plot(X, y_fit, '-', lw=2)
        plt.plot(X, y_fit_low, 'r--', lw=1)
        plt.plot(X, y_fit_upp, 'r--', lw=1)
        #plt.plot(X, predict_mean_ci_low, 'maroon', lw=1)
        #plt.plot(X, predict_mean_ci_upp, 'maroon', lw=1)
        plt.xticks(fontsize=8)
        plt.yticks(fontsize=8)
        plt.xlabel('Total Reads/cell', fontsize = 8)
        plt.ylabel('Gene Counts', fontsize = 8)
        plt.show()

    orig_len = len (adata.obs)
    libSize = adata.X.sum(axis=1).A1
    adata = adata[libSize >= minLSize, :]
    libSize = adata.X.sum(axis=1).A1 # total reads/cell
    
    genelst = [var.upper() for var in adata.var_names]
    mtGenes = [g.startswith('MT-') for g in genelst]
    nGenes = (adata.X != 0).sum(axis=1).A1 # of expressed genes/cell
    
    model = sm.OLS(nGenes, libSize) # (y, X)
    results = model.fit()
    #print(results.summary())
    _, data, _ = summary_table(results, alpha = a)
    fit_values = data[:, 2]
    predict_ci_low, predict_ci_upp = data[:, 6:8].T

    if len(mtGenes) > 0:
        mtCounts = adata.X[:, mtGenes].sum(axis=1).A1 # total reads of mt-genes/cell
        mtProportion = mtCounts/libSize
        mt_model = sm.OLS(mtCounts, libSize) # (y, X)
        mt_results = mt_model.fit()
        _, mt_data, _ = summary_table(mt_results, alpha = a)
        mt_fit_values = mt_data[:, 2]
        mt_predict_ci_low, mt_predict_ci_upp = mt_data[:, 6:8].T
        if plot:
            plotQC(libSize, mtCounts, mt_fit_values, mt_predict_ci_low, mt_predict_ci_upp)
            
        mask = (mtCounts > mt_predict_ci_low) & (mtCounts < mt_predict_ci_upp) & \
                         (nGenes > predict_ci_low) & (nGenes < predict_ci_upp) &  \
                         (libSize < 2 * np.mean(libSize))
    else:
        if plot:
            plotQC(libSize, nGenes, fit_values, predict_ci_low, predict_ci_upp)
            
        mask = (nGenes > predict_ci_low) & (nGenes < predict_ci_upp) &  \
                         (libSize < 2 * np.mean(libSize))
    
    adata = adata[mask, :]
    print ('Filtered out {} cells...'.format(orig_len - len (adata.obs))) 
    
    return adata
    
#example: 
#import urllib.request
#code = 'https://raw.githubusercontent.com/yjgeno/py_public/main/scTools/scQC.py'
#data = urllib.request.urlopen(code).read()
#exec(data)

#pdmc = sc.read_10x_mtx('./hg19/', var_names='gene_symbols', cache=True) 
#scQC(pdmc, mtThreshold = 0.1, minLSize = 1000, a = 0.05, plot = True)

