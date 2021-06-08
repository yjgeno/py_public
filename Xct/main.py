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
    receptors = LR['receptor'].str.split('_', expand=True)
    receptors.columns = ['rec_A', 'rec_B', 'rec_C']
    LRs = pd.concat([LR[['pathway_name', 'ligand']], receptors], axis=1)
    del LR
    return LRs


def XctInfo(ada, CellA, CellB, permute = False, verbose = False):
    result = {}
    AB = ada[ada.obs['ident'].isin([CellA, CellB]), :]
    
    if permute:
        #np.random.seed(42)
        labels_pmt = np.random.permutation(AB.obs['ident'])
        AB.obs['ident'] = labels_pmt
        if verbose:
            print('Cell A and B permutated')
    #print(AB.obs['ident'][:30].unique(), AB.obs['ident'][-30:].unique()) #check           
    A = AB[AB.obs['ident'] == CellA, :]
    B = AB[AB.obs['ident'] == CellB, :]
    #print(A.shape, B.shape)
   
    l_exp = []
    l_var = []
    for l in LRs['ligand']:
        if l not in np.array(A.var_names):
            l_exp.append(np.nan)
            l_var.append(np.nan)
        else:
            l_exp.append(np.mean(A[:, A.var_names.isin([l])].X))
            l_var.append(np.var(A[:, A.var_names.isin([l])].X))
    result['l_exp'] = l_exp
    result['l_var'] = l_var
    
    for rec in ['rec_A', 'rec_B', 'rec_C']:
        r_exp = []
        r_var = []
        for r in LRs[rec]:
            if r not in np.array(B.var_names):
                r_exp.append(np.nan)
                r_var.append(np.nan)
            else:
                r_exp.append(np.mean(B[:, B.var_names.isin([r])].X)) #mean expression of L and R
                r_var.append(np.var(B[:, B.var_names.isin([r])].X))
        result['{}_exp'.format(rec)] = r_exp
        result['{}_var'.format(rec)] = r_var
    
    return result
 
    
def XctSelection(dict_AB, IsPmt = False, verbose = False): # if input is permutated dataset
    LRs_Xct = LRs.copy()  
    LRs_Xct = pd.concat([LRs_Xct, pd.DataFrame.from_dict(dict_AB)], axis=1)
    
    mask1 = np.invert(LRs_Xct[['l_exp', 'rec_A_exp']].isna().any(axis=1)) # remove NA
    LRs_Xct = LRs_Xct[mask1]
    if not IsPmt:
        mask2 = (LRs_Xct['l_exp'] > 0) & (LRs_Xct['rec_A_exp'] > 0) # remove 0 for original LR
        LRs_Xct = LRs_Xct[mask2]
    if verbose:
        print('Selected {} LR pairs'.format(LRs_Xct.shape[0]))
    
    LRs_Xct['rec_exp'] = LRs_Xct[['rec_A_exp', 'rec_B_exp', 'rec_C_exp']].max(axis=1) #mean for R
    LRs_Xct['rec_var'] = LRs_Xct[['rec_A_var', 'rec_B_var', 'rec_C_var']].max(axis=1) #var for R
     
    return LRs_Xct


def XctScore1(LRs_Xct):
    LRs_Xct['LR_score'] = LRs_Xct['l_exp'] * LRs_Xct['rec_exp']
    return LRs_Xct['LR_score'].to_numpy(dtype=float)


def XctScore2(LRs_Xct):
    LRs_Xct['LR_score2'] = (LRs_Xct['l_exp']**2 + LRs_Xct['l_var']) * (LRs_Xct['rec_exp']**2 + LRs_Xct['rec_var'])  
    return LRs_Xct['LR_score2'].to_numpy(dtype=float)


def XctScores(CellA, CellB, LRs, func, n=100): #func: score method, permute n times
    scores = []
    for _ in range(n):
        p = XctSelection(XctInfo(CellA, CellB, permute = True), IsPmt = True, verbose = False)
        scores.append(func(p.loc[list(set(p.index) & set(LRs.index))])) #filter
    assert all(len(i) == len(orig_score) for i in scores) #check if equal len of selected LR pairs
    return np.array(scores).T  #transpose for further looping


def Xct_PermuTest(orig_score, scores, p = 0.05):
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


def vis(orig_score, scores, i, LRs = LRs_Selected, density = False): #index i in LRs_Selected
    print('LR pair: {} - {}'.format(LRs.iloc[i]['ligand'], LRs.iloc[i]['rec_A']))
    plt.hist(scores[i], density = density)
    plt.axvline(x = orig_score[i], color = 'r')
    plt.show()
    
#LRs_Selected = XctSelection(XctInfo(ada, CellA, CellB, permute = False), IsPmt = False, verbose = True) #selected original LR pairs
#orig_score = XctScore1(LRs_Selected)  
#scores = XctScores(CellA, CellB, LRs_Selected, XctScore1, n=10)
#enriched, pvals, counts = Xct_PermuTest(orig_score, scores)
#LRs_Enriched = LRs_Selected.iloc[enriched, :] #enriched pairs

#vis(orig_score, scores, 5, density = False)








    
    
    
    
