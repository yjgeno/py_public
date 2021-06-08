import urllib.request
code = urllib.request.urlopen('https://raw.githubusercontent.com/yjgeno/py_public/main/Xct/main.py').read()
exec(code)

#adata built
global LRs 
LRs = Xct_DB()

def Xct(ada, CellA, CellB, n=10, verbose = True):
    print(CellA, CellB)
    
    LRs_ref = Xct_init(ada, CellA, CellB, DB = LRs, verbose = False)
    orig_Scores = Xct_Score(df = LRs_ref)
    #np.random.seed(1)
    Scores = Xct_Scores(ada, CellA, CellB, LRs_ref = LRs_ref, n=n)
    enriched, _, counts = Xct_PmtTest(orig_Scores, Scores)
    #LRs_Enriched = LRs_ref.iloc[enriched, :]
    if verbose:
        print('{} out of {} L-R pairs enriched'.format(len(enriched), len(LRs_ref)))

    return sum(orig_Scores[enriched])
  
#test
#Xct(ada, 'Inflam. FIB', 'Inflam. DC', n = 10)

Cells = ada.obs['ident'].unique()
results = {}
for CellA in Cells:
    for CellB in Cells:
        results[CellA + ' - ' + CellB] = Xct(ada, CellA, CellB)
