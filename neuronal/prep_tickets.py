import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import math
import datetime
import subprocess

#
# 
#

def ticket_by_date(date, tickets, params):
    
    df_tickets = pd.read_csv(tickets, sep='##OS##', engine='python')
    df_tickets.columns = ['id', 'date_de_creation', 'date_de_restoration', 'date_de_fix', 'description_du_ticket', 'date_commentaire', 'commentaires', 'label', 'nidt_noeud', 'elem', 'status_code', 'status']
    
    df_params = pd.read_csv(params, sep=';', engine='python')
    df_params.columns = ['Field', 't', 'techno', 'date', 'Ur', 'no_ur', 'omc', 'code_dr', 'nom_dr', 'Plaque', 'zone_MKT', 'nom_omc', 'mfs', 'nom_pcu', 'bsc', 'rnc', 'no_rnc', 'ni_rnc', 'site', 'no_bsc', 'ni_bsc', 'no_site', 'ni_site', 'no_secteur', 'secteur', 'no_bts', 'bts', 'lac', 'ci', 'nidt', 'no_noeud', 'nidt_noeud', 'ni', 'date_fn8', 'date_mest', 'Eteint', 'nci', 'locked', 'barred', 'ms_tx_pwr_max_cch', 'ncc', 'bcc', 'bcch', 'nodeb', 'nodebid', 'gprs', 'edge_actif', 'bts_power', 'nb_pdtch_statiques', 'max_pkt_radio_type', 'nb_pdch_tot', 'nb_pdch_stat', 'sac', 'hsdpa', 'hsupa', 'TGV', 'ni_msc', 'nom_msc', 'FreqDownlink', 'tdb', 'nidt_msc', 'Template', 'Bande', 'cid', 'nb_bandes', 'objQoS', 'zone_ARCEP', 'txt_bandeau', 'do', 'nom_secteur']

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
    df_joined = df_tickets.merge(df_params, left_on='nidt_noeud', right_on='nidt_noeud', how='inner')

    # df = df_joined[df_joined['date_de_creation'] == date]

    # if df.empty:
    #     raise Exception("There is no known ticket at that date")
    if df_joined.empty:
        return
    return df_joined


# print(ticket_by_date("\"2020-01-24 22:34:00\"", 'bertrand_03_2020.csv', 'osiris_params.csv'))
# print(ticket_by_date("\"2020-01-24 22:34:00\"", 'bertrand_04_2020.csv', 'osiris_params.csv'))
# print(ticket_by_date("\"2020-01-24 22:34:00\"", 'bertrand_05_2020.csv', 'osiris_params.csv'))
# print(ticket_by_date("\"2020-01-24 22:34:00\"", 'bertrand_06_2020.csv', 'osiris_params.csv'))
# print(ticket_by_date("\"2020-01-24 22:34:00\"", 'bertrand_07_2020.csv', 'osiris_params.csv'))
# print(ticket_by_date("\"2020-01-24 22:34:00\"", 'bertrand_08_2020.csv', 'osiris_params.csv'))
# print(ticket_by_date("\"2020-01-24 22:34:00\"", 'bertrand_09_2020.csv', 'osiris_params.csv'))
# print(ticket_by_date("\"2020-01-24 22:34:00\"", 'bertrand_10_2020.csv', 'osiris_params.csv'))
# print(ticket_by_date("\"2020-01-24 22:34:00\"", 'bertrand_11_2019.csv', 'osiris_params.csv'))
print(ticket_by_date("\"2020-01-24 22:34:00\"", 'bertrand_12_2019.csv', 'osiris_params.csv'))