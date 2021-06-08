import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scanpy as sc
from anndata import AnnData

sc.settings.verbosity = 0
# adata object built
# for Cell A, B


def Xct_DB(specis = 'Human'):
    if specis == 'mouse':
        LR = pd.read_csv('https://raw.githubusercontent.com/yjgeno/Ligand-Receptor-Pairs/master/Mouse/Mouse-2020-Jin-LR-pairs.csv')
    else:
        LR = pd.read_csv('https://raw.githubusercontent.com/yjgeno/Ligand-Receptor-Pairs/master/Human/Human-2020-Jin-LR-pairs.csv')
    ligands = LR['ligand'].str.split('_', expand=True)
    ligands.columns = ['lig_A', 'lig_B']
    receptors = LR['receptor'].str.split('_', expand=True)
    receptors.columns = ['rec_A', 'rec_B', 'rec_C']
    LRs = pd.concat([LR[['pathway_name']], ligands, receptors], axis=1)
    del LR
    
    return LRs


def Xct_init(ada, CellA, CellB, DB, verbose = False):
    result = {}
    AB = ada[ada.obs['ident'].isin([CellA, CellB]), :]
    A = AB[AB.obs['ident'] == CellA, :]
    B = AB[AB.obs['ident'] == CellB, :]
    
    for lig in ['lig_A', 'lig_B']:
        l_exp = []
        l_var = []
        for l in DB[lig]:
            if l not in np.array(A.var_names):
                l_exp.append(np.nan)
                l_var.append(np.nan)
            else:
                l_exp.append(np.mean(A[:, A.var_names.isin([l])].X))
                l_var.append(np.var(A[:, A.var_names.isin([l])].X))
        result['{}_exp'.format(lig)] = l_exp
        result['{}_var'.format(lig)] = l_var
    
    for rec in ['rec_A', 'rec_B', 'rec_C']:
        r_exp = []
        r_var = []
        for r in DB[rec]:
            if r not in np.array(B.var_names):
                r_exp.append(np.nan)
                r_var.append(np.nan)
            else:
                r_exp.append(np.mean(B[:, B.var_names.isin([r])].X)) #mean expression of L and R
                r_var.append(np.var(B[:, B.var_names.isin([r])].X))
        result['{}_exp'.format(rec)] = r_exp
        result['{}_var'.format(rec)] = r_var
    #print(result.keys())

    LRs_Xct = DB.copy() 
    LRs_Xct = pd.concat([LRs_Xct, pd.DataFrame.from_dict(result)], axis=1)
    mask1 = np.invert(LRs_Xct[['lig_A_exp', 'rec_A_exp']].isna().any(axis=1)) # remove NA
    mask2 = (LRs_Xct['lig_A_exp'] > 0) & (LRs_Xct['rec_A_exp'] > 0) # remove 0 for original LR
    LRs_Xct = LRs_Xct[mask1 & mask2]
    if verbose:
        print('Selected {} LR pairs'.format(LRs_Xct.shape[0]))

    return LRs_Xct
        

def Xct_pmt(ada, CellA, CellB, Ref, verbose = False):
    result = {}
    AB = ada[ada.obs['ident'].isin([CellA, CellB]), :].copy()
    
    #np.random.seed(42)
    labels_pmt = np.random.permutation(AB.obs['ident'])
    AB.obs['ident'] = labels_pmt         
    A = AB[AB.obs['ident'] == CellA, :]
    B = AB[AB.obs['ident'] == CellB, :]
    
    if verbose:
        print('Cell A and B permutated')       
       #print(AB.obs['ident'][:10].unique(), AB.obs['ident'][-10:].unique()) #check
        print('# of {}: {}; # of {}: {}'.format(CellA, A.shape[0], CellB, B.shape[0]))
   
    for lig in ['lig_A', 'lig_B']:
        l_exp = []
        l_var = []
        for l in Ref[lig]:
            if l not in np.array(A.var_names):
                l_exp.append(0)
                l_var.append(0)
            else:
                l_exp.append(np.mean(A[:, A.var_names.isin([l])].X))
                l_var.append(np.var(A[:, A.var_names.isin([l])].X))
        result['{}_exp'.format(lig)] = l_exp
        result['{}_var'.format(lig)] = l_var
    
    for rec in ['rec_A', 'rec_B', 'rec_C']:
        r_exp = []
        r_var = []
        for r in Ref[rec]:
            if r not in np.array(B.var_names):
                r_exp.append(0)
                r_var.append(0)
            else:
                r_exp.append(np.mean(B[:, B.var_names.isin([r])].X)) #mean expression of L and R
                r_var.append(np.var(B[:, B.var_names.isin([r])].X))
        result['{}_exp'.format(rec)] = r_exp
        result['{}_var'.format(rec)] = r_var
    
    #print(result.keys()) 
    result = pd.DataFrame.from_dict(result)
    assert len(result) == len(Ref)
    
    return result
  

def Xct_Score(df, method = 0):
    lig = ['lig_A_exp', 'lig_B_exp', 'lig_A_var', 'lig_B_var']
    rec = ['rec_A_exp', 'rec_B_exp', 'rec_C_exp', 'rec_A_var', 'rec_B_var', 'rec_C_var']
    if (set(lig) | set(rec)).issubset(df.columns):
        exp_R = np.array(df[rec[:3]].max(axis=1)) #mean for R   
        exp_L = np.array(df[lig[:2]].max(axis=1))    
        if method == 0:
            S = exp_L * exp_R        
        elif method == 1:
            var_R = np.array(df[rec[3:]].max(axis=1)) #var for R
            var_L = np.array(df[lig[2:]].max(axis=1))
            S = (exp_L**2 + var_L)*(exp_R**2 + var_R)
        
        return S.astype(float)
    
    else:
        print('Error: column names')


def Xct_Scores(ada, CellA, CellB, LRs_ref, s = 0, n = 100): #s: score method, permute n times
    scores = []
    for _ in range(n):
        LRs_pmt = Xct_pmt(ada, CellA, CellB, Ref = LRs_ref, verbose = False)
        scores.append(Xct_Score(df = LRs_pmt, method = s))

    assert all(len(i) == len(LRs_ref) for i in scores) #check if equal len of ref LR pairs
    return np.array(scores).T  #transpose for further looping


def Xct_PmtTest(orig_score, scores, p = 0.05):
    enriched_i = []
    pvals = []
    counts = []
    for i, dist in enumerate(scores):
        count = sum(orig_score[i] > value for value in dist)
        pval = 1- count/len(dist)
        pvals.append(pval)
        counts.append(count)
        
        if pval < p:
            enriched_i.append(i)           
    
    return enriched_i, pvals, counts


def vis(orig_Scores, Scores, i, LRs, density = False): #index i in LRs_Selected
    print('LR pair: {} - {}'.format(LRs.iloc[i]['lig_A'], LRs.iloc[i]['rec_A']))
    plt.hist(Scores[i], density = density)
    plt.axvline(x = orig_Scores[i], color = 'r')
    plt.show()


    
#test
#CellA = 'Inflam. FIB'
#CellB = 'Inflam. DC'
#LRs = Xct_DB()
#LRs_ref = Xct_init(ada, CellA, CellB, DB = LRs, verbose = True)
#orig_Scores = Xct_Score(df = LRs_ref)
#np.random.seed(1)
#Scores = Xct_Scores(ada, CellA, CellB, LRs_ref = LRs_ref, n=10)
#enriched, pvals, counts = Xct_PmtTest(orig_Scores, Scores)
#LRs_Enriched = LRs_ref.iloc[enriched, :]
#vis(orig_Scores, Scores, 0, LRs = LRs_ref)



    
    
    
    
