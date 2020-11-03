import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import math
import datetime
import subprocess
import itertools
#
# 
#

def ticket_by_date(date, tickets, params, variables):
    

    # Dataframe tickets
    df_tickets = pd.read_csv(tickets, sep='##OS##', engine='python')
    df_tickets.columns = ['id', 'date_de_creation', 'date_de_restoration', 'date_de_fix', 'description_du_ticket', 'date_commentaire', 'commentaires', 'label', 'nidt_noeud', 'elem', 'status_code', 'status']
    
    # Dataframe params
    df_params = pd.read_csv(params, sep=';', engine='python')
    df_params.columns = ['Field', 't', 'techno', 'date', 'Ur', 'no_ur', 'omc', 'code_dr', 'nom_dr', 'Plaque', 'zone_MKT', 'nom_omc', 'mfs', 'nom_pcu', 'bsc', 'rnc', 'no_rnc', 'ni_rnc', 'site', 'no_bsc', 'ni_bsc', 'no_site', 'ni_site', 'no_secteur', 'secteur', 'no_bts', 'bts', 'lac', 'ci', 'nidt', 'no_noeud', 'nidt_noeud', 'ni', 'date_fn8', 'date_mest', 'Eteint', 'nci', 'locked', 'barred', 'ms_tx_pwr_max_cch', 'ncc', 'bcc', 'bcch', 'nodeb', 'nodebid', 'gprs', 'edge_actif', 'bts_power', 'nb_pdtch_statiques', 'max_pkt_radio_type', 'nb_pdch_tot', 'nb_pdch_stat', 'sac', 'hsdpa', 'hsupa', 'TGV', 'ni_msc', 'nom_msc', 'FreqDownlink', 'tdb', 'nidt_msc', 'Template', 'Bande', 'cid', 'nb_bandes', 'objQoS', 'zone_ARCEP', 'txt_bandeau', 'do', 'nom_secteur']

    # Dataframe variables
    df_variables = pd.read_csv(variables, sep=',', engine='python')

    # Dataframe tickets + params
    df_joined = pd.merge(df_params, df_tickets, how='outer', on='nidt_noeud')
    df_joined = df_joined.fillna(0)


    #####################################################################################################################

    # df_double_joined = df_joined.merge(df_variables, left_on='ni', right_on='ni', how='outer')
    # df = df_joined[df_joined['date_de_creation'] == date]

    # if df.empty:
    #     raise Exception("There is no known ticket at that date")
    if df_joined.empty:
        return
    return df_joined


def ticket_by_date_chunks(date, tickets, params, variables):
    

    # Dataframe tickets
    df_tickets = pd.read_csv(tickets, sep='##OS##', engine='python', chunksize=50000)
    df_tickets.columns = ['id', 'date_de_creation', 'date_de_restoration', 'date_de_fix', 'description_du_ticket', 'date_commentaire', 'commentaires', 'label', 'nidt_noeud', 'elem', 'status_code', 'status']
    
    # Dataframe params
    df_params = pd.read_csv(params, sep=';', engine='python', chunksize=50000)
    df_params.columns = ['Field', 't', 'techno', 'date', 'Ur', 'no_ur', 'omc', 'code_dr', 'nom_dr', 'Plaque', 'zone_MKT', 'nom_omc', 'mfs', 'nom_pcu', 'bsc', 'rnc', 'no_rnc', 'ni_rnc', 'site', 'no_bsc', 'ni_bsc', 'no_site', 'ni_site', 'no_secteur', 'secteur', 'no_bts', 'bts', 'lac', 'ci', 'nidt', 'no_noeud', 'nidt_noeud', 'ni', 'date_fn8', 'date_mest', 'Eteint', 'nci', 'locked', 'barred', 'ms_tx_pwr_max_cch', 'ncc', 'bcc', 'bcch', 'nodeb', 'nodebid', 'gprs', 'edge_actif', 'bts_power', 'nb_pdtch_statiques', 'max_pkt_radio_type', 'nb_pdch_tot', 'nb_pdch_stat', 'sac', 'hsdpa', 'hsupa', 'TGV', 'ni_msc', 'nom_msc', 'FreqDownlink', 'tdb', 'nidt_msc', 'Template', 'Bande', 'cid', 'nb_bandes', 'objQoS', 'zone_ARCEP', 'txt_bandeau', 'do', 'nom_secteur']

    # Dataframe variables
    df_variables = pd.read_csv(variables, sep=',', engine='python', chunksize=50000)


    for tickets, params, variables in zip(df_tickets, df_params, df_variables):
        tickets.columns = ['id', 'date_de_creation', 'date_de_restoration', 'date_de_fix', 'description_du_ticket', 'date_commentaire', 'commentaires', 'label', 'nidt_noeud', 'elem', 'status_code', 'status']
        params.columns = ['Field', 't', 'techno', 'date', 'Ur', 'no_ur', 'omc', 'code_dr', 'nom_dr', 'Plaque', 'zone_MKT', 'nom_omc', 'mfs', 'nom_pcu', 'bsc', 'rnc', 'no_rnc', 'ni_rnc', 'site', 'no_bsc', 'ni_bsc', 'no_site', 'ni_site', 'no_secteur', 'secteur', 'no_bts', 'bts', 'lac', 'ci', 'nidt', 'no_noeud', 'nidt_noeud', 'ni', 'date_fn8', 'date_mest', 'Eteint', 'nci', 'locked', 'barred', 'ms_tx_pwr_max_cch', 'ncc', 'bcc', 'bcch', 'nodeb', 'nodebid', 'gprs', 'edge_actif', 'bts_power', 'nb_pdtch_statiques', 'max_pkt_radio_type', 'nb_pdch_tot', 'nb_pdch_stat', 'sac', 'hsdpa', 'hsupa', 'TGV', 'ni_msc', 'nom_msc', 'FreqDownlink', 'tdb', 'nidt_msc', 'Template', 'Bande', 'cid', 'nb_bandes', 'objQoS', 'zone_ARCEP', 'txt_bandeau', 'do', 'nom_secteur']
        
        # Dataframe tickets + params
        df_joined = pd.merge(tickets, params, how='outer', on='nidt_noeud')
        df_joined = df_joined.fillna(0)

        df_double_joined = pd.merge(df_joined, variables, how='outer', on='ni')

    ret = "Everything went well"
    return ret


print(ticket_by_date_chunks("\"2020-01-24 22:34:00\"", 'bertrand_03_2020.csv', 'osiris_params.csv', 'data_h.csv'))