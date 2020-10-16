# and parallelize the process. 
#

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import math


# ----------------------------------------------
# ----------------------------------------------

#get every number that is not NaN
def test(L):
    ret = []
    line = 0
    for j in L:
        column = 0
        for i in j:
            if not math.isnan(i):
                ret.append((line, column, i))
            column += 1
        line += 1 
    return ret


#Format the output (variable 1, variable 2, coefficient of correlation)
def translate(L, columns):
    ret = []
    for i in L:
        if (i[0] != i[1]):
            ret.append((columns[i[0]], columns[i[1]], i[2]))
    return ret

# Return values that seems to be correlated with the "issue"
def get_cause(issue, table):
    ret = []
    for (a,b,c) in table:
        if a == issue:
            ret.append((a,b,c))
    
    ret = filter(lambda x: x[2] > 0.9 or x[2] < -0.9, ret)
    return sorted(list(ret), key=lambda tup: tup[2], reverse=True)
    
    
    #sortde_table = ret.sort(reverse=True)
    #size = len(ret)
    
    # Value still needs to be determined, starting with 30%
    # Could maybe be higher than a certain percentage, to be determined also
    #percentage = 20
    #cut = math.floor(percentage * size / 100)
    
    #sorted_list = sorted(ret, key=lambda tup: tup[2], reverse=True)
    #return sorted_list[:cut] + sorted_list[-cut:] 
    #return sorted_list

def values_by_date(date, issue):
    chunklist = []
    flag = False
    df = None
    for chunk in df_chunk:
        if flag:
            break
        df = chunk[chunk["date_debut_mesure"] == "2020-08-31"]
    
        if not df.empty:
            flag = True
            corr = df.corr(method='pearson')
            tabs = corr.values
            chunk_list = chunk_list + translate(test(tabs), chunk.columns)

    return chunklist


df_chunk = pd.read_csv('ia_nokia4gj2.csv', sep=',', chunksize=10000)
chunk_list = []

for chunk in df_chunk:
    corr = chunk.corr(method='pearson')
    tabs = corr.values
    chunk_list = chunk_list + translate(test(tabs), chunk.columns)
    
#print(chunk_list)
causes = get_cause("sql_ne", chunk_list)
print(causes)

def correlated_variables(L):
    ret = []
    for (a, b, c) in L:
        ret.append(b)
    return ret

#print(correlated_variables(causes))