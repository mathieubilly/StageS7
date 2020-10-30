import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import math
import datetime
import subprocess


def ticket_by_date(date, file):
    df_chunk = pd.read_csv(file, sep='##OS##', engine='python')

    df_chunk.columns = ['id', 'date_de_creation', 'date_de_restoration', 'date_de_fix', 'description_du_ticket', 'date_commentaire', 'commentaires', 'label', 'code_elem', 'elem', 'status_code', 'status']
    
    '''
    flag = False
    for chunk in df_chunk:
        if flag:
            break
        df = chunk[chunk['date_de_creation'] == date]

        if not df.empty:
            flag = True
            return df
    '''

    df = df_chunk[df_chunk['date_de_creation'] == date]

    if df.empty:
        raise Exception("There is no known ticket at that date")
    return df


print(ticket_by_date("\"2020-01-24 22:34:00\"", 'bertrand_01_2020.csv'))