import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scanpy as sc
from anndata import AnnData
import scipy
from scipy.optimize import least_squares
import time
import progressbar

sc.settings.verbosity = 0
# adata object built
# for Cell A, B

def Xct_DB(specis = 'Human'):
    if specis == 'Mouse':
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
  
def get_metric(ada, verbose = False): #require normalized data
    #sc.pp.normalize_total(ada, target_sum=1e6)
    if isinstance(ada.X, scipy.sparse.csr.csr_matrix):
        data_norm = csr_matrix.toarray(ada.X)
    else:
        data_norm = ada.X
    
    if verbose:
        print('(cell, feature):', data_norm.shape)
    
    mean = np.mean(data_norm, axis = 0)
    var = np.var(data_norm, axis = 0)
    mean[mean == 0] = 1e-12
    dispersion = var / mean    
    cv = np.sqrt(var) / mean
    
    return mean, var, dispersion, cv
  
def chen2016_fit(mean, cv, plot = False, verbose = False): 
    xdata_orig = mean #raw
    ydata_orig = np.log10(cv) #log   
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
        print('{} (for fit) / {} (filtered -Inf) / {} (original) features for the fit'.format(ix-cdx[0], len(ydata), len(ydata_orig)))
        
    #lst fit
    def residuals(coeff, t, y):
        return y - 0.5 * (np.log10(coeff[1]/t + coeff[0])) # x: raw mean y:log(cv)

    x0 = np.array([0, 1], dtype=float) # initial guess a=0, b=1
    model = least_squares(residuals, x0, loss='soft_l1', f_scale= 0.01, args=(xSeq, ySeq_log))
     
    def predict_robust(coeff, x):
        return 0.5 * (np.log10(coeff[1]/x + coeff[0]))
    
    ydataFit = predict_robust(model.x, xdata_orig) #logCV
    
    def cv_diff(obs_cv, fit_cv): 
        #diff = ((np.log10(obs_cv) - fit_cv) / fit_cv) * 100 # bio_var/tech_var, if use np.log10(cv): risk of -Inf
        #diff = obs_cv - (10**fit_cv) #raw count space
        diff = np.log10(obs_cv - (10**fit_cv))
        
        #print(np.isinf(diff).any())
        #pd.DataFrame([bio_cv, fit_cv, bio_cv / fit_cv]).to_csv('./check_score.csv', header = None)
        return {key: v for key, v in zip(ada.var_names, diff)} # dict key: gene name
          
    if plot:
        y_predict = predict_robust(model.x, xSeq) 
        plt.figure(figsize=(6, 5), dpi=80)
        
        plt.scatter(np.log10(xdata), ydata, s=3, marker='o') # orig
        plt.plot(np.log10(xSeq), ySeq_log, c='black', label='poly fit') # poly fit
        plt.plot(np.log10(xSeq), y_predict, label='robust lsq', c='r') # robust nonlinear
        
        #ind = list(res[res['padj'] < fdr].index)[:ngenes] # index for filtered xdata, ydata
        #for n, i in zip(HVG, ind):
        #    plt.annotate(n, xy = (np.log10(xdata)[i], ydata[i]), xytext = (np.log10(xdata)[i]+1, ydata[i]+0.5),
        #                 arrowprops=dict(arrowstyle="->", connectionstyle="arc3"))
        plt.xlabel('log10(mean)')
        plt.ylabel('log10(CV)')
        plt.legend(loc='lower left')
        plt.show()
          
    return cv_diff(cv, ydataFit) #log CV difference
  
att = ['exp', 'var', 'disp', 'CV', 'CV_res']  
def Xct_init(ada, CellA, CellB, DB = LRs, verbose = False):
    result = {}
    AB = ada[ada.obs['ident'].isin([CellA, CellB]), :].copy()
    A = AB[AB.obs['ident'] == CellA, :].copy()
    B = AB[AB.obs['ident'] == CellB, :].copy()
    
    mean_A, _, _, cv_A = get_metric(A)
    mean_B, _, _, cv_B = get_metric(B)
    
    cv_res_A = chen2016_fit(mean_A, cv_A)
    cv_res_B = chen2016_fit(mean_B, cv_B)

    
    for lig in ['lig_A', 'lig_B']:
        l_exp, l_var, l_disp, l_CV, l_CV_res = ([] for i in range(len(att)))
        for l in DB[lig]:
            if l not in np.array(A.var_names):
                [x.append(value) for x, value in zip([l_exp, l_var, l_disp, l_CV, l_CV_res], [np.nan]*len(att)*len(DB))] 
                #append att-long nan

            else:
                l_counts = A[:, A.var_names.isin([l])].copy()
                mean, var, dispersion, cv = get_metric(l_counts)
                [x.append(value) for x, value in zip([l_exp, l_var, l_disp, l_CV, l_CV_res], 
                                                     [np.round(mean[0],11), var[0], dispersion[0], cv[0], cv_res_A[l]])] 
                #append 4 times, round 11: convert 1e-12 to 0
          
        
        for l_result, a in zip([l_exp, l_var, l_disp, l_CV, l_CV_res], att):
            result['{}_{}'.format(lig, a)] = l_result
    
    for rec in ['rec_A', 'rec_B', 'rec_C']:
        r_exp, r_var, r_disp, r_CV, r_CV_res = ([] for i in range(len(att)))
        for r in DB[rec]:
            if r not in np.array(B.var_names):
                [x.append(value) for x, value in zip([r_exp, r_var, r_disp, r_CV, r_CV_res], [np.nan]*len(att)*len(DB))]

            else:
                r_counts = B[:, B.var_names.isin([r])].copy()
                mean, var, dispersion, cv = get_metric(r_counts)
                [x.append(value) for x, value in zip([r_exp, r_var, r_disp, r_CV, r_CV_res], 
                                                     [np.round(mean[0],11), var[0], dispersion[0], cv[0], cv_res_B[r]])]           
        
        for r_result, a in zip([r_exp, r_var, r_disp, r_CV, r_CV_res], att):
            result['{}_{}'.format(rec, a)] = r_result  
    print(result.keys())
    
    LRs_Xct = DB.reset_index(drop = True, inplace = False) 
    LRs_Xct = pd.concat([LRs_Xct, pd.DataFrame.from_dict(result)], axis=1) # concat 1:1 since sharing same index
    #mask1 = np.invert(LRs_Xct[['lig_A_exp', 'rec_A_exp']].isna().any(axis=1)) # remove NA
    mask1 = (LRs_Xct['lig_A_exp'] > 0) & (LRs_Xct['rec_A_exp'] > 0) # filter 0 for first LR
    LRs_Xct = LRs_Xct[mask1]
    
    pattern_orig = LRs_Xct[['lig_A','lig_B', 'rec_A', 'rec_B', 'rec_C']].isnull()
    pattern_obs = LRs_Xct[['lig_A_exp','lig_B_exp', 'rec_A_exp','rec_B_exp', 'rec_C_exp']].isnull()
    mask2 = (pattern_orig.values == pattern_obs.values).all(axis=1) # for LR complex
    LRs_Xct = LRs_Xct[mask2]
    
    if verbose:
        print('Selected {} LR pairs'.format(LRs_Xct.shape[0]))
 
    return LRs_Xct #df
    

def Xct_pmt(ada, CellA, CellB, Ref, verbose = False):
    result = {}  
    AB = ada[ada.obs['ident'].isin([CellA, CellB]), :].copy()
    AB_raw = AB
    #np.random.seed(42)
    
    if CellA == CellB:
        AB = AB.concatenate(AB)
        AB_index = np.random.permutation(AB.shape[0]) #locally pmt AB, not globally
        t = np.array([AB.X[i] for i in AB_index])
        AB.X = t
        del t
        A = AB[:AB.shape[0]//2, :].copy()
        B = AB[AB.shape[0]//2:, :].copy()
    else:
        labels_pmt = np.random.permutation(AB.obs['ident'])
        AB.obs['ident'] = labels_pmt    
        A = AB[AB.obs['ident'] == CellA, :].copy()
        B = AB[AB.obs['ident'] == CellB, :].copy()
    
    if verbose:
        assert not (AB_raw[:A.shape[0], :].X == A.X).all() #check if permutated
        del AB_raw
        print('Cell A and B permutated')       
       #print(AB.obs['ident'][:10].unique(), AB.obs['ident'][-10:].unique()) #check
        print('# of {}: {}; # of {}: {}'.format(CellA, A.shape[0], CellB, B.shape[0]))
        
    mean_A, _, _, cv_A = get_metric(A)
    mean_B, _, _, cv_B = get_metric(B)
    
    cv_res_A = chen2016_fit(mean_A, cv_A, plot = False)
    cv_res_B = chen2016_fit(mean_B, cv_B, plot = False)
        
    for lig in ['lig_A', 'lig_B']:
        l_exp, l_var, l_disp, l_CV, l_CV_res = ([] for i in range(len(att)))
        for l in Ref[lig]:
            if l not in np.array(A.var_names):
                [x.append(value) for x, value in zip([l_exp, l_var, l_disp, l_CV, l_CV_res], [np.nan]*len(att)*len(Ref))]
            else:
                l_counts = A[:, A.var_names.isin([l])].copy()
                mean, var, dispersion, cv = get_metric(l_counts)
                [x.append(value) for x, value in zip([l_exp, l_var, l_disp, l_CV, l_CV_res], 
                                                     [np.round(mean[0],11), var[0], dispersion[0], cv[0], cv_res_A[l]])]

        for l_result, a in zip([l_exp, l_var, l_disp, l_CV, l_CV_res], att):
            result['{}_{}'.format(lig, a)] = l_result
    
    
    for rec in ['rec_A', 'rec_B', 'rec_C']:
        r_exp, r_var, r_disp, r_CV, r_CV_res = ([] for i in range(len(att)))
        for r in Ref[rec]:
            if r not in np.array(B.var_names):
                [x.append(value) for x, value in zip([r_exp, r_var, r_disp, r_CV, r_CV_res], [np.nan]*len(att)*len(Ref))]
            else:
                r_counts = B[:, B.var_names.isin([r])].copy()
                mean, var, dispersion, cv = get_metric(r_counts)
                [x.append(value) for x, value in zip([r_exp, r_var, r_disp, r_CV, r_CV_res], 
                                                     [np.round(mean[0],11), var[0], dispersion[0], cv[0], cv_res_B[r]])]           
        
        for r_result, a in zip([r_exp, r_var, r_disp, r_CV, r_CV_res], att):
            result['{}_{}'.format(rec, a)] = r_result  
    
    
    #print(result.keys()) 
    LRs_pmt = pd.DataFrame.from_dict(result)
    assert len(LRs_pmt) == len(Ref)
    

    return LRs_pmt #df
  
  
def Xct_info(df): #integrate LR complex
    start_i = (df.columns).tolist().index('lig_A_exp') #6 for ref, 0 for pmt
    exp_i, var_i, disp_i, CV_i, CV_res_i = ([] for i in range(len(att)))
    for atts, res in zip([exp_i, var_i, disp_i, CV_i, CV_res_i], range(len(att))):
        for i, name in enumerate(df.columns[start_i:]): 
            if i%len(att) == res:
                atts.append(name)
        #print(atts)
        #raise KeyError('check column names')
    #output: exp_i == ['lig_A_exp', 'lig_B_exp', 'rec_A_exp', 'rec_B_exp', 'rec_C_exp']
    
    result = {}
    for atts, a in zip([exp_i, var_i, disp_i, CV_i, CV_res_i], att): 
        lig = df[atts[:2]]
        rec = df[atts[2:]]
        result['{}_L'.format(a)] = np.asarray(np.power(lig.prod(axis=1), 1./lig.notna().sum(1))) #L geomean: Kth root
        result['{}_R'.format(a)] = np.asarray(np.power(rec.prod(axis=1), 1./rec.notna().sum(1))) #R
    #print(result.keys()) 
    #['exp_L', 'exp_R', 'var_L', 'var_R', 'disp_L', 'disp_R', 'CV_L', 'CV_R']

    return result #dict
  
  
def Score(result, method = 0, a = 1):
    S0 = result['exp_L'] * result['exp_R'] 
    S0 /= np.percentile(S0, 80) 
    S0 = S0/(0.5 + S0)
 
    if method == 0:
        return S0 
    if method == 1:
        S = (result['exp_L']**2 + a*result['var_L'])*(result['exp_R']**2 + a*result['var_R'])
        S = S/(0.5 + S)
    if method == 2:
        S = result['disp_L'] * result['disp_R']
    if method == 3:
        S = result['CV_L'] + a*result['CV_R']
    if method == 4:
        #print(result['CV_res_L'])
        #print(result['CV_res_R'])
        result['CV_res_L'][result['CV_res_L'] < 0] = 0
        S = abs(result['CV_res_L'] * result['CV_res_R'])
        S = S/(0.5+S) + a*S0
        #S = S0 / (S/(0.5+S))
    if method == 5:
        S = result['CV_res_L'] * result['CV_res_R']
        
    return S #.astype(float)
  

def Xct_Scores(ada, CellA, CellB, LRs_ref, method = 0, a = 1, n = 100, progress = True): #s: score method, permute n times
    scores = []
    
    if not progress:
        for _ in range(n):
            LRs_pmt = Xct_pmt(ada, CellA, CellB, Ref = LRs_ref, verbose = False)
            scores.append(Score(Xct_info(LRs_pmt), method = method, a = a))
    else:
        for _, _ in zip(progressbar.progressbar(range(n)), range(n)):
            LRs_pmt = Xct_pmt(ada, CellA, CellB, Ref = LRs_ref, verbose = False)
            scores.append(Score(Xct_info(LRs_pmt), method = method, a = a))
            #time.sleep(0.02)

    assert all(len(i) == len(LRs_ref) for i in scores) #check if equal len of ref LR pairs
    return np.array(scores).T  #transpose for further looping
  
  
  
def Xct_PmtTest(orig_score, scores, p = 0.05):
    enriched_i, pvals, counts = ([] for _ in range(3))
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
    
    
    
    
    
