
import keras
from keras.utils import to_categorical
from keras.models import Sequential
from keras.utils import np_utils
from keras.wrappers.scikit_learn import KerasClassifier
from keras.layers.core import Dense, Activation, Dropout
from keras.optimizers import SGD

#import sklearn
#from sklearn.model_selection import train_test_split
#from sklearn.metrics import mean_squared_error

#import opencv2
import pandas as pd
import numpy as np
import math

def write_matrix(m):
    f = open("output.txt", "a")
    for i in m:
        for j in i:
            f.write(" ")
            f.write(str(j))
    
    f.close()

def check_matrix(M):
    ret = []
    tmp = M[0][0]
    for i in M:
        for j in i:
            if j != tmp:
                print(j)
                ret.append(j)
    return len(ret) 

def sublists(l):
    res = []
    for i in l:
        res.append([i])
    return res

def ticket_by_date_chunks(date, tickets, params, variables):
    
    chunks = 50000
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
   
    full_data = full_data.apply(pd.to_numeric, errors='coerce')
    full_data = full_data.fillna(1)
    #full_data.apply(pd.to_numeric)

    print('Full data concatenated')
    #labels_2 = list(range(0, len(labels)))
    #labels_2 = sublists(labels_2)
    


    #print(y_train.shape[0])

    print('Every String replaced by 1 and passed to numeric')        

    set_size = 149000
    full_data_matrix = full_data.values
    #X_train = full_data_matrix[0:set_size]
   
    len_data = len(full_data_matrix)
 
    X_train = full_data_matrix[:len_data - 1000]
    labels_2 = np.random.randint(38, size=(set_size, 1))

    X_test = full_data_matrix[len_data - 1000:]
    ytest = keras.utils.to_categorical(labels_2, num_classes=38)

    y_train = keras.utils.to_categorical(labels_2, num_classes=38)
    
    #X_train = np.random.random((1000,20))
    #X_test = np.random.random((100,20))

    print('data splited')

    print('labels created')
    input_dim = X_train.shape[1]

    nb_classes = 38
    #print('first shape X: ', X_train.shape[0])
    #print('first shjape Y: ', ytrain.shape[0])
    
    print('preprocessing data')
  
    #print(X_train.shape)
    print('model created')
    
    model = Sequential()
    model.add(Dense(256, activation='tanh', input_dim=input_dim))
    model.add(Dropout(0.5))
    model.add(Dense(256, activation='tanh'))
    model.add(Dropout(0.5))
    model.add(Dense(256, activation='tanh'))
    model.add(Dropout(0.5))
    model.add(Dense(nb_classes, activation='softmax'))
    
    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    #print("Xtrain dims ar [", X_train.shape[0],X_train.shape[1],"]")     
    #print("ytrain dims ar [", ytrain.shape[0],ytrain.shape[1],"]")  
   
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

    #res = model.predict_on_batch(X_train)   
    #write_matrix(res)

    #print(check_matrix(X_train[80000:81000]))    
    
    model.fit(X_train, y_train, epochs=200, batch_size=64)
    #score = model.evaluate(X_test, ytest, batch_size=128)      

    ynew = model.predict_classes(X_test)

    #print(X_test)
    return ynew

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
