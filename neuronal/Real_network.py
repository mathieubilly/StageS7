from keras.models import Sequential
from keras.utils import np_utils
from keras.layers.core import Dense, Activation, Dropout
from sklearn.model_selection import train_test_split

import pandas as pd
import numpy as np

def ticket_by_date_chunks(date, tickets, params, variables):
    
    chunks = 25000
    # Dataframe tickets
    df_tickets = pd.read_csv(tickets, sep='##OS##', engine='python', chunksize=chunks)
    df_tickets.columns = ['id', 'date_de_creation', 'date_de_restoration', 'date_de_fix', 'description_du_ticket', 'date_commentaire', 'commentaires', 'label', 'nidt_noeud', 'elem', 'status_code', 'status']
    
    
    # Dataframe params
    df_params = pd.read_csv(params, sep=';', engine='python', chunksize=chunks)
    df_params.columns = ['Field', 't', 'techno', 'date', 'Ur', 'no_ur', 'omc', 'code_dr', 'nom_dr', 'Plaque', 'zone_MKT', 'nom_omc', 'mfs', 'nom_pcu', 'bsc', 'rnc', 'no_rnc', 'ni_rnc', 'site', 'no_bsc', 'ni_bsc', 'no_site', 'ni_site', 'no_secteur', 'secteur', 'no_bts', 'bts', 'lac', 'ci', 'nidt', 'no_noeud', 'nidt_noeud', 'ni', 'date_fn8', 'date_mest', 'Eteint', 'nci', 'locked', 'barred', 'ms_tx_pwr_max_cch', 'ncc', 'bcc', 'bcch', 'nodeb', 'nodebid', 'gprs', 'edge_actif', 'bts_power', 'nb_pdtch_statiques', 'max_pkt_radio_type', 'nb_pdch_tot', 'nb_pdch_stat', 'sac', 'hsdpa', 'hsupa', 'TGV', 'ni_msc', 'nom_msc', 'FreqDownlink', 'tdb', 'nidt_msc', 'Template', 'Bande', 'cid', 'nb_bandes', 'objQoS', 'zone_ARCEP', 'txt_bandeau', 'do', 'nom_secteur']

    # Dataframe variables
    df_variables = pd.read_csv(variables, sep=',', engine='python', chunksize=chunks)

    labels = []
    chunk_list = []
    for tickets, params, variables in zip(df_tickets, df_params, df_variables):
        tickets.columns = ['id', 'date_de_creation', 'date_de_restoration', 'date_de_fix', 'description_du_ticket', 'date_commentaire', 'commentaires', 'label', 'nidt_noeud', 'elem', 'status_code', 'status']
        params.columns = ['Field', 't', 'techno', 'date', 'Ur', 'no_ur', 'omc', 'code_dr', 'nom_dr', 'Plaque', 'zone_MKT', 'nom_omc', 'mfs', 'nom_pcu', 'bsc', 'rnc', 'no_rnc', 'ni_rnc', 'site', 'no_bsc', 'ni_bsc', 'no_site', 'ni_site', 'no_secteur', 'secteur', 'no_bts', 'bts', 'lac', 'ci', 'nidt', 'no_noeud', 'nidt_noeud', 'ni', 'date_fn8', 'date_mest', 'Eteint', 'nci', 'locked', 'barred', 'ms_tx_pwr_max_cch', 'ncc', 'bcc', 'bcch', 'nodeb', 'nodebid', 'gprs', 'edge_actif', 'bts_power', 'nb_pdtch_statiques', 'max_pkt_radio_type', 'nb_pdch_tot', 'nb_pdch_stat', 'sac', 'hsdpa', 'hsupa', 'TGV', 'ni_msc', 'nom_msc', 'FreqDownlink', 'tdb', 'nidt_msc', 'Template', 'Bande', 'cid', 'nb_bandes', 'objQoS', 'zone_ARCEP', 'txt_bandeau', 'do', 'nom_secteur']

        # labels = tickets['label']
        for word in tickets.label.unique():
            if word not in labels:
                labels.append(word)

        # Dataframe tickets + params
        df_joined = pd.merge(tickets, params, how='outer', on='nidt_noeud')
        df_joined = df_joined.fillna(0)

        # Merged 3 datasets
        df_double_joined = pd.merge(df_joined, variables, how='outer', on='ni')

        chunk_list.append(df_double_joined)

    full_data = pd.concat(chunk_list)


    print('Full data concatenated')        
    
    # Splitting train set and test set
    
    X_train, X_test = train_test_split(full_data)
    # X_test = X_test.values.astype(float32)
    # X_train = X_train.values.astype(float32)

print(ticket_by_date_chunks("\"2020-01-24 22:34:00\"", 'bertrand_03_2020.csv', 'osiris_params.csv', 'ia_nokia4gj2.csv'))

'''
    print('data splited')
    # Create labels
    y_train = np_utils.to_categorical(labels)

    print('labels created')
    scale = np.max(X_train)
    X_train /= scale
    X_test /= scale

    mean = np.std(X_train)
    X_train -= mean
    X_test -= mean

    input_dim = X_train.shape[1]
    nb_classes = y_train.shape[1]

    print('preprocessing data')

    model = Sequential()
    model.add(Dense(128, input_dim=input_dim))
    model.add(Activation('relu'))
    model.add(Dropout(0.15))
    model.add(Dense(128))
    model.add(Activation('relu'))
    model.add(Dropout(0.15))
    model.add(Dense(nb_classes))
    model.add(Activation('softmax'))
   
    print('model created')

   # we'll use categorical xent for the loss, and RMSprop as the optimizer
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

    print("Training...")
    history = model.fit(X_train, y_train, epochs=10, batch_size=16, validation_split=0.1, verbose=2)

print(ticket_by_date_chunks("\"2020-01-24 22:34:00\"", 'bertrand_01_2020.csv', 'osiris_params.csv', 'ia_nokia4gj2.csv'))



# Read the preproccessed data 

train = pd.read_csv('train.csv')
labels = train.iloc[:,0].values.astype('int32')
X_train = (train.iloc[:,1:].values).astype('float32')
X_test = (pd.read_csv('test.csv').values).astype('float32')


# Convert list of labels to binary class matrix

y_train = np_utils.to_categorical(labels) 

# Pre-processing: divide by max and substract mean



# Here's a Deep Dumb MLP (DDMLP)

model = Sequential()
model.add(Dense(128, input_dim=input_dim))
model.add(Activation('relu'))
model.add(Dropout(0.15))
model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dropout(0.15))
model.add(Dense(nb_classes))
model.add(Activation('softmax'))

# we'll use categorical xent for the loss, and RMSprop as the optimizer
model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

print("Training...")
history = model.fit(X_train, y_train, epochs=10, batch_size=16, validation_split=0.1, verbose=2)

print("Generating test predictions...")
preds = model.predict_classes(X_test, verbose=0)

print(preds)


def write_preds(preds, fname):
    pd.DataFrame({"ImageId": list(range(1,len(preds)+1)), "Label": preds}).to_csv(fname, index=False, header=True)

write_preds(preds, "keras-mlp.csv")
'''