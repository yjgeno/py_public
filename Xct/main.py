import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scanpy as sc
from anndata import AnnData
import scipy
from scipy.optimize import least_squares
import warnings
warnings.filterwarnings("ignore")
sc.settings.verbosity = 0

#require adata with layer 'raw' (counts) and 'log1p' (normalized), cell labels in obs 'idents'

class Xct_metrics():
    __slots__ = ('genes', 'DB', '_genes_index_DB')
    def __init__(self, adata, specis = 'Human'): #adata: cellA * allgenes
        self.genes = adata.var_names
        self.DB = self.Xct_DB()
        self._genes_index_DB = self.get_index(DB = self.DB)

    
    def Xct_DB(self, specis = 'Human'):
        if specis == 'Mouse':
            LR = pd.read_csv('https://raw.githubusercontent.com/yjgeno/Ligand-Receptor-Pairs/master/Mouse/Mouse-2020-Jin-LR-pairs.csv')
        if specis == 'Human':
            LR = pd.read_csv('https://raw.githubusercontent.com/yjgeno/Ligand-Receptor-Pairs/master/Human/Human-2020-Jin-LR-pairs.csv')
        ligands = LR['ligand'].str.split('_', expand=True)
        ligands.columns = ['lig_A', 'lig_B']
        receptors = LR['receptor'].str.split('_', expand=True)
        receptors.columns = ['rec_A', 'rec_B', 'rec_C']
        LRs = pd.concat([LR[['pathway_name']], ligands, receptors], axis=1)
        del LR

        return LRs
    
    def subset(self):
        genes = np.ravel(self.DB.iloc[:, 1:].values) #['lig_A', 'lig_B', 'rec_A', 'rec_B', 'rec_C']
        genes = np.unique(genes[genes != None])
        genes_use = self.genes.intersection(genes)
            
        return [list(self.genes).index(g) for g in genes_use]    #index in orig adata
    
    def get_index(self, DB):
        g_LRs = DB.iloc[:, 1:6].values #['lig_A', 'lig_B', 'rec_A', 'rec_B', 'rec_C']
        gene_list = [None] + list(self.genes) #LR genes intersect with DB

        gene_index = np.zeros(len(np.ravel(g_LRs)), dtype = int)
        for g in gene_list:
            g_index = np.asarray(np.where(np.isin(np.ravel(g_LRs), g)))
            if g_index.size == 0:
                continue
            else:
                for i in g_index:
                    gene_index[i] = gene_list.index(g) 
        genes_index_DB = np.array(gene_index).reshape(g_LRs.shape) #gene index refer to subset adata var + 1
        
        return genes_index_DB


    def get_metric(self, adata, verbose = False): #require normalized data
        data_norm = scipy.sparse.csr_matrix.toarray(adata.X) if scipy.sparse.issparse(adata.X) else adata.X
        if verbose:
            print('(cell, feature):', data_norm.shape)
        
        if (data_norm % 1 != 0).any(): #check space: True for log (float), False for counts (int)
            mean = np.mean(data_norm, axis = 0)
            var = np.var(data_norm, axis = 0)
            mean[mean == 0] = 1e-12
            dispersion = var / mean    
            cv = np.sqrt(var) / mean

            return mean, var, dispersion, cv
        else:
            raise ValueError("require log data")
    
    def chen2016_fit(self, adata, plot = False, verbose = False): #require raw data 
        data_raw = adata.layers['raw'] #.copy()
        if (data_raw % 1 != 0).any():
            raise ValueError("require counts (int) data")
        else:
            mean_raw = np.mean(data_raw, axis = 0)
            var_raw = np.var(data_raw, axis = 0)
            mean_raw[mean_raw == 0] = 1e-12
            cv_raw = np.sqrt(var_raw) / mean_raw
        
        xdata_orig = mean_raw #raw
        ydata_orig = np.log10(cv_raw) #log   
        rows = len(xdata_orig) #features

        r = np.invert(np.isinf(ydata_orig)) # filter -Inf
        ydata = ydata_orig[r] #Y
        xdata = xdata_orig[r] #X

        #poly fit: log-log
        z = np.polyfit(np.log10(xdata), ydata, 2) 

        def predict(z, x):
            return z[0]*(x**2) + z[1]*x + z[2]

        xSeq_log = np.arange(min(np.log10(xdata)), max(np.log10(xdata)), 0.005) 
        ySeq_log = predict(z, xSeq_log)  #predicted y

        #start point for fit
        #plt.hist(np.log10(xdata), bins=100)
        def h(i):
            a = np.log10(xdata) >= (xSeq_log[i] - 0.05)
            b = np.log10(xdata) < (xSeq_log[i] + 0.05)
            return np.sum((a & b))

        gapNum = [h(i) for i in range(0, len(xSeq_log))] #density histogram of xdata
        cdx = np.nonzero(np.array(gapNum) > rows*0.005)[0] #start from high density bin

        xSeq = 10 ** xSeq_log 

        #end pointy for fit
        yDiff = np.diff(ySeq_log, 1) #a[i+1] - a[i]
        ix = np.nonzero((yDiff > 0) & (np.log10(xSeq[0:-1]) > 0))[0] # index of such (X, Y) at lowest Y

        if len(ix) == 0:
            ix = len(ySeq_log) - 1 # use all
        else:
            ix = ix[0]

        #subset data for fit
        xSeq_all = 10**np.arange(min(np.log10(xdata)), max(np.log10(xdata)), 0.001) 
        xSeq = xSeq[cdx[0]:ix]
        ySeq_log = ySeq_log[cdx[0]:ix]

        if verbose:
            #print(ix, cdx[0])
            print('{} (intervals for fit) / {} (filtered -Inf) / {} (original) features for the fit'.format(ix-cdx[0], len(ydata), len(ydata_orig)))

        #lst fit
        def residuals(coeff, t, y):
            return y - 0.5 * (np.log10(coeff[1]/t + coeff[0])) # x: raw mean y:log(cv)

        x0 = np.array([0, 1], dtype=float) # initial guess a=0, b=1
        model = least_squares(residuals, x0, loss='soft_l1', f_scale= 0.01, args=(xSeq, ySeq_log))

        def predict_robust(coeff, x):
            return 0.5 * (np.log10(coeff[1]/x + coeff[0]))

        ydataFit = predict_robust(model.x, xdata_orig) #logCV

        def cv_diff(obs_cv, fit_cv): 
            obs_cv[obs_cv == 0] = 1e-12
            diff = np.log10(obs_cv) - fit_cv
            return diff #{key: v for key, v in zip(self.genes, diff)} 

        if plot:
            y_predict = predict_robust(model.x, xSeq) 
            plt.figure(figsize=(6, 5), dpi=80)   
            plt.scatter(np.log10(xdata), ydata, s=3, marker='o') # orig
            plt.plot(np.log10(xSeq), ySeq_log, c='black', label='poly fit') # poly fit
            plt.plot(np.log10(xSeq), y_predict, label='robust lsq', c='r') # robust nonlinear

            #ind = list(res[res['padj'] < fdr].index)[:ngenes] # index for filtered xdata, ydata
            #for n, i in zip(['CCL19', 'CCR7', 'CXCL12', 'CXCR4'], [371, 388, 592, 598]):
            #   plt.annotate(n, xy = (np.log10(xdata)[i], ydata[i]), xytext = (np.log10(xdata)[i]+1, ydata[i]+0.5),
            #                 arrowprops=dict(arrowstyle="->", connectionstyle="arc3"))
            plt.xlabel('log10(mean)')
            plt.ylabel('log10(CV)')
            plt.legend(loc='lower left')
            plt.show()
        
        #obs_cv = self.cv.copy()
        diff = cv_diff(cv_raw, ydataFit)
        return diff #log CV difference
      
      
      
class Xct(Xct_metrics):

    def __init__(self, adata, CellA, CellB, pmt = False):
        Xct_metrics.__init__(self, adata)
        self._metric_names = ['mean', 'var', 'disp', 'cv', 'cv_res']
        ada_A = adata[adata.obs['ident'] == CellA, :].copy()
        ada_B = adata[adata.obs['ident'] == CellB, :].copy()
        
        self._metric_A = np.vstack([self.get_metric(ada_A), self.chen2016_fit(ada_A)]) #len 5
        self._metric_B = np.vstack([self.get_metric(ada_B), self.chen2016_fit(ada_B)])
        
        if not pmt:
            self.ref = self.fill_metric()
            self.genes_index = self.get_index(DB = self.ref)
       
        del ada_A, ada_B
               
    def fill_metric(self, ref_obj = None, verbose = False):
        if ref_obj is None:
            genes_index = self._genes_index_DB
        else:
            #if isinstance(ref_obj, Xct):
            genes_index = ref_obj.genes_index
        #print(genes_index)
        
        index_L = genes_index[:, :2]
        index_R = genes_index[:, 2:]

        df = pd.DataFrame()

        for metric_A, metric_B, metric in zip(self._metric_A, self._metric_B, self._metric_names):
            filled_L = []
            filled_R = []
            for i in np.ravel(index_L):
                if i == 0:
                    filled_L.append(0) #none expression
                else:
                    filled_L.append(np.round(metric_A[i-1], 11))
            filled_L = np.array(filled_L, dtype=float).reshape(index_L.shape)

            for i in np.ravel(index_R):
                if i == 0:
                    filled_R.append(0)
                else:
                    filled_R.append(np.round(metric_B[i-1], 11))
            filled_R = np.array(filled_R, dtype=float).reshape(index_R.shape)

            filled = np.concatenate((filled_L, filled_R), axis=1)
            result = pd.DataFrame(data = filled, columns = [f'{metric}_L1', f'{metric}_L2', 
                                                             f'{metric}_R1', f'{metric}_R2', f'{metric}_R3'])
            df = pd.concat([df, result], axis=1)
        
        #DB = skin.DB.reset_index(drop = True, inplace = False) 
        
        
        if ref_obj is None:
            df = pd.concat([self.DB, df], axis=1) # concat 1:1 since sharing same index
            mask1 = (df['mean_L1'] > 0) & (df['mean_R1'] > 0) # filter 0 for first LR
            df = df[mask1]

            pattern_orig = df.iloc[:, 1:6].isnull() #L-R complex
            pattern_obs = df.iloc[:, 6:11].isin([0]) #mean expression
            mask2 = (pattern_orig.values == pattern_obs.values).all(axis=1) # for LR complex
            df = df[mask2]
            
        else: 
            ref_DB = self.DB.iloc[ref_obj.ref.index, :].reset_index(drop = True, inplace = False) #match index
            df = pd.concat([ref_DB, df], axis=1)
            df.set_index(pd.Index(ref_obj.ref.index), inplace = True)
            
        
        df.replace(to_replace={0:None}, inplace = True) #for geo mean
        
        for i, name in zip(range(6, 31, 5), self._metric_names):
            lig = df.iloc[:, i:i+2]
            rec = df.iloc[:, i+2:i+5]
            if i < 26:
                df[f'{name}_L'] = np.asarray(np.power(lig.prod(axis=1), 1./lig.notna().sum(1))) #L geomean: Kth root
                df[f'{name}_R'] = np.asarray(np.power(rec.prod(axis=1), 1./rec.notna().sum(1))) #R
            else:
                df[f'{name}_L'] = np.asarray(lig.sum(axis = 1, skipna = True))
                df[f'{name}_R'] = np.asarray(rec.sum(axis = 1, skipna = True))
        
        #df.to_csv('df.csv', index=False)
        if verbose:
            print('Selected {} LR pairs'.format(df.shape[0]))

        return df
    
    
    def score(self, ref_DB = None, method = 0, a = 1):
        if ref_DB is None:
            ref_DB = self.ref.copy()
        S0 = ref_DB['mean_L'] * ref_DB['mean_R'] 
        S0 /= np.percentile(S0, 80) 
        S0 = S0/(0.5 + S0)

        if method == 0:
            return S0  
        if method == 1:
            S = (ref_DB['mean_L']**2 + a*ref_DB['var_L'])*(ref_DB['mean_R']**2 + a*ref_DB['var_R'])
            S = S/(0.5 + S)
        if method == 2:
            S = ref_DB['disp_L'] * ref_DB['disp_R']
        if method == 3:
            S = ref_DB['cv_L'] + a*ref_DB['cv_R']
        if method == 4:
            ref_DB['cv_res_L'][ref_DB['cv_res_L'] < 0] = 0
            S = abs(ref_DB['cv_res_L'] * ref_DB['cv_res_R'])
            S = S/(0.5+S) + a*S0
            #S = S0 / (S/(0.5+S))
        if method == 5:
            S = ref_DB['cv_res_L'] + a*ref_DB['cv_res_R']

        return S #.astype(float)
      
      
def scores(adata, ref_obj, method = 1, n = 100):
    result = []
    temp = adata.copy()
    
    for _ in range(n):
        labels_pmt = np.random.permutation(temp.obs['ident']) #pmt gloablly
        temp.obs['ident'] = labels_pmt
        #ada_pmt = pmt(adata)
        pmt_obj = Xct(temp, 'Inflam. FIB', 'Inflam. DC', pmt =True)
        df_pmt = pmt_obj.fill_metric(ref_obj = ref_obj)
        result.append(pmt_obj.score(ref_DB = df_pmt, method = method))
    
    return np.array(result).T
  
  
def pmt_test(orig_score, scores, p = 0.05):
    enriched_i, pvals, counts = ([] for _ in range(3))
    for i, dist in enumerate(scores):
        count = sum(orig_score[i] > value for value in dist)
        pval = 1- count/len(dist)
        pvals.append(pval)
        counts.append(count)
        
        if pval < p:
            enriched_i.append(i)           
    
    return enriched_i, pvals, counts
  
  
#test
#s1 = Xct(ada, 'Inflam. FIB', 'Inflam. DC')
#df1 = s1.fill_metric()


