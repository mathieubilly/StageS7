# and parallelize the process. 
#

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import math
import datetime
import subprocess

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


def by_month(date, values, issue):
    
    chunk_list = []
    start_date = date
    end_date = (datetime.datetime.strptime(date, '%Y-%m-%d') - datetime.timedelta(days=30)).strftime("%Y-%m-%d")

    df = values[(values['date_debut_mesure'] > end_date) & (values['date_debut_mesure'] <= start_date)]

    #print(df)
    corr = df.corr(method='pearson')
    tabs = corr.values
    chunk_list = chunk_list + translate(test(tabs), df.columns)
    return get_cause(issue, chunk_list)


# Get the correlation coefficent list for each variable, by cell, sorted 
def by_cell(data, cell, month_analysis, date=None):
    chunk_list = []
    df = None
    for chunk in data:
        df = chunk[chunk["ni"] == cell]
        if not df.empty:
            corr = df.corr(method='pearson')
            tabs = corr.values
            chunk_list = chunk_list + translate(test(tabs), chunk.columns)
            chunk_list = filter(lambda x: x[2] > 0.9 or x[2] < -0.9, chunk_list)
            return sorted(chunk_list, key=lambda tup: tup[2], reverse=True)

    raise Exception("Cell not found")


# Get the correlation coefficent list for each variable, by group of cells, sorted 
def by_group(data, ni_enodeb, month_analysis, date=None):
    chunk_list = []
    df = None
    for chunk in data:
        df = chunk[chunk["ni_enodeb"] == ni_enodeb]
        if not df.empty:
            corr = df.corr(method='pearson')
            tabs = corr.values
            chunk_list = chunk_list + translate(test(tabs), chunk.columns)
            chunk_list = filter(lambda x: x[2] > 0.9 or x[2] < -0.9, chunk_list)
            return sorted(chunk_list, key=lambda tup: tup[2], reverse=True)

    raise Exception("Cell not found")

# Goes through the data , does the coreraltion matrix and return only the interesting ones 
def data_treatment(data, issue):
    chunk_list = []
    for chunk in data:
        corr = chunk.corr(method='pearson')
        tabs = corr.values
        chunk_list = chunk_list + translate(test(tabs), chunk.columns)

    #print(chunk_list)
    return get_cause(issue, chunk_list)
    #print(causes)

def correlated_variables(L):
    ret = []
    for (a, b, c) in L:
        ret.append(b)
    return ret

#print(correlated_variables(by_month("2019-09-07", df_not_chunk)))

def diagnostic(name, issue, month_analysis, date = None):
    if not name.endswith('.csv'):
        subprocess.call(['mv', name, name + '.csv'])

    df = None
    if (month_analysis):
        if date == None:
            raise Exception('You must enter a valid date')
        df = pd.read_csv(name, sep=',')
        return correlated_variables(by_month(date, df, issue))
    else:
        df = pd.read_csv(name, sep=',', chunksize=100000)
        return correlated_variables(data_treatment(df, issue))

# print(diagnostic('ia_nokia4gj2.csv', "sql_ne", False))
print(by_cell(pd.read_csv('ia_nokia4gj2.csv', sep=',', chunksize=1000), 138121509, False))
# print(by_group(pd.read_csv('ia_nokia4gj2.csv', sep=',', chunksize=1000), 42236, False))