# and parallelize the process. 
#

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import math
import datetime


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

def values_by_date(date, issue, df_chunk):
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


def by_month(date, values):
    
    chunk_list = []
    start_date = date
    end_date = (datetime.datetime.strptime(date, '%Y-%m-%d') - datetime.timedelta(days=30)).strftime("%Y-%m-%d")

    df = values[(values['date_debut_mesure'] > end_date) & (values['date_debut_mesure'] <= start_date)]

    #print(df)
    corr = df.corr(method='pearson')
    tabs = corr.values
    chunk_list = chunk_list + translate(test(tabs), df.columns)
    return get_cause("sql_ne", chunk_list)


    #for chunk in values: 
    #    #df = datetime.datetime.strptime(chunk[chunk["date_debut_mesure"]], "%Y-%m-%d").month == month# & datetime.datetime.strptime(chunk["date_debut_mesure"], "%Y-%m-%d").year == year]
    #    #if df is not None:
    #    #flag = True
    #    chunk['date_debut_mesure'] = pd.to_datetime(chunk['date_debut_mesure'])
    #    mask = (chunk['date_debut_mesure'] > start_date) & (chunk['date_debut_mesure'] <= end_date)
    #    df = chunk.loc[mask]
    #    
    #    df = df[(df['date'] > '2000-6-1') & (df['date'] <= '2000-6-10')]
    #    if not df.empty
    #        corr = df.corr(method='pearson')
    #        tabs = corr.values
    #        chunk_list = chunk_list + translate(test(tabs), chunk.columns)
    #        return chunk_list
#
    #return chunk_list




#df_chunk = pd.read_csv('ia_nokia4gj2.csv', sep=',', chunksize=10000)

df_not_chunk = pd.read_csv('ia_nokia4gj2.csv', sep=',')

def data_treatment(data):
    chunk_list = []
    for chunk in data:
        corr = chunk.corr(method='pearson')
        tabs = corr.values
        chunk_list = chunk_list + translate(test(tabs), chunk.columns)

    #print(chunk_list)
    causes = get_cause("sql_ne", chunk_list)
    #print(causes)

def correlated_variables(L):
    ret = []
    for (a, b, c) in L:
        ret.append(b)
    return ret

print(correlated_variables(by_month("2019-09-07", df_not_chunk)))