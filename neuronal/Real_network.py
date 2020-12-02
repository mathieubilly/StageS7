
from keras.utils import to_categorical
from keras.models import Sequential
from keras.utils import np_utils
from keras.wrappers.scikit_learn import KerasClassifier
from keras.layers.core import Dense, Activation, Dropout

#import sklearn
#from sklearn.model_selection import train_test_split
#from sklearn.metrics import mean_squared_error

#import opencv2
import pandas as pd
import numpy as np
import math

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
        df_joined = df_joined.fillna(1)

        # Merged 3 datasets
        df_double_joined = pd.merge(df_joined, variables, how='outer', on='ni')

        chunk_list.append(df_double_joined)
        break

    full_data = pd.concat(chunk_list)

    print("Data is concaneted and ready to be replaced")
    #full_data = np.where(type(full_data) == str, 1, full_data)
    #full_data.replace(to_replace="(\"?[0-9]*[a-zA-Z '()]+[,.]*([0-9]*[a-zA-Z '.,()]*)*\"?)|(\"?([0-9]*[-:][0-9]*)+\"?)", value="1", regex=True)
    #full_data.replace(to_replace="(\"?([0-9]*[-:][0-9]*)+\"?)", value=r"\1", regex=True, inplace=True)
   
    full_data = full_data.apply(pd.to_numeric, errors='coerce')
    full_data.fillna(1)
    #full_data.apply(pd.to_numeric)

    print('Full data concatenated')
    labels_2 = list(range(0, len(labels)))
    y_train = np_utils.to_categorical(labels_2, num_classes=len(labels_2))

    print('Every String replaced by 1 and passed to numeric')        
    
    # Splitting train set and test set

    X_train = full_data.iloc[0:4000]
    X_test = full_data.iloc[4567:5000]
    labels_2 = list(range(0,len(labels)))
    #X_train = full_data[0:4000]
    #X_test = full_data[4001:4500]

    #, X_test, y_train, y_test = train_test_split(full_data, labels_2, test_size=0.30, random_state=40)

    #ytrain = to_categorical(y_train)
    #y_test = to_categorical(y_test)

    # X_test = X_test.values.astype(float32)
    # X_train = X_train.values.astype(float32)

    print('data splited')
    # Create labels

    #ytrain = labels_2
    ytrain = np_utils.to_categorical(labels_2)    

    print('labels created')
    #X_train = np_utils.normalize(X_train)
    input_dim = X_train.shape[1]
    #print(input_dim)
    #print(X_train)
    nb_classes = ytrain.shape[1]

    print(nb_classes)
    #print('first shape X: ', X_train.shape[0])
    #print('first shjape Y: ', ytrain.shape[0])
    
    print('preprocessing data')
    #X_train = X_train[0:38]
    X_train = X_train.fillna(1)
    #ytrain = ytrain.reshape(1, 38, 2)   
    print(X_train.shape)
    print('model created')
    
    model = Sequential()
    model.add(Dense(256, input_dim=input_dim))
    model.add(Activation('relu'))
    model.add(Dropout(0.15))
    model.add(Dense(256))
    model.add(Activation('relu'))
    model.add(Dropout(0.15))
    model.add(Dense(nb_classes))
    model.add(Activation('softmax'))
    
    print("Xtrain dims ar [", X_train.shape[0],X_train.shape[1],"]")     
    print("ytrain dims ar [", ytrain.shape[0],ytrain.shape[1],"]")  
   
    #we'll use categorical xent for the loss, and RMSprop as the optimizer
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    print("Training...")
    return model.predict_on_batch(X_train)   
#model.fit(X_train, ytrain, epochs=20, batch_size=16, validation_split=0.1, verbose=2)
    
print(ticket_by_date_chunks("\"2020-01-24 22:34:00\"", 'bertrand_03_2020.csv', 'osiris_params.csv', 'ia_nokia4gj2.csv'))

'''

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
