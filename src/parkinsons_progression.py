from loadAndPreprocess import LoadAndPreprocess
from DataProccessing import DLNetwork
from enum import Enum

protein_train_path = "parkinsons-disease-progression-prediction\\amp-parkinsons-disease-progression-prediction\\train_proteins.csv"
peptide_train_path = "parkinsons-disease-progression-prediction\\amp-parkinsons-disease-progression-prediction\\train_peptides.csv"
train_clinical_data ="parkinsons-disease-progression-prediction\\amp-parkinsons-disease-progression-prediction\\train_clinical_data.csv"

class Mode(Enum):
    GRU = 1
    FULLY_CONNECTED = 2
    TWO_INPUT_FC_LSTM = 3
    MULTI_INPUT_LSTM = 4


loadInstance = LoadAndPreprocess(protein_train_path, peptide_train_path, train_clinical_data)
udprs_visits_vector , updrs_visits, visit_updrs_dict = loadInstance.GetUdprsData()
visit_protein_dict = loadInstance.GetProteinInputDataset()
peptide_visits_vector , peptide_visits = loadInstance.GetPeptideData()

#create dictionaries
peptide_dict = dict(zip(peptide_visits, peptide_visits_vector))
udprs_dict = dict(zip(updrs_visits, udprs_visits_vector))
common_visits = list(set(updrs_visits).intersection(peptide_visits))

dlNetwork = DLNetwork(common_visits,udprs_dict,peptide_dict)


mode = Mode.FULLY_CONNECTED


if(mode==Mode.MULTI_INPUT_LSTM):
    updrsPerPatient_train, updrsPerPatient_train_labels, updrsPerPatient_Test, updrsPerPatient_Test_labels, proteins_train, proteins_test, peptides_train, peptides_test = loadInstance.GetProteinDatasets(visit_protein_dict)
    multi_input_model = dlNetwork.build_3_input_network(len(updrsPerPatient_train[0]),len(proteins_train[0]))
    multi_input_history = multi_input_model.fit([updrsPerPatient_train,proteins_train,peptides_train],updrsPerPatient_train_labels,epochs=900, batch_size=32)
    dlNetwork.plotLoss(multi_input_history)
    multi_input_model.predict([updrsPerPatient_Test,proteins_test,peptides_test])
    updrsPerPatient_Test_labels
#################################################################################################
##  This network Predict updrs for Visit_60 both by priod progression of updrs and peptide abundance
################################################################################################

if(mode==Mode.TWO_INPUT_FC_LSTM):
    updrsPerPatient_train,updrsPerPatient_train_labels, updrsPerPatient_Test,updrsPerPatient_Test_labels = loadInstance.GetUpdrsPerPatient(updrs_visits)
    peptide_train,peptide_tests = loadInstance.GetpeptidePerLastVisit()
    multi_input_model = dlNetwork.build_multi_input_network(len(updrsPerPatient_train[0]))
    multi_input_history = multi_input_model.fit([updrsPerPatient_train,peptide_train],updrsPerPatient_train_labels,epochs=900, batch_size=32)
    dlNetwork.plotLoss(multi_input_history)
    multi_input_model.predict([updrsPerPatient_Test,peptide_tests])
    updrsPerPatient_Test_labels
###################################################################################
##  This network Predict updrs for Visit_60 by prior parkinson updrs progression V_0,V_3.....V_48
##################################################################################

if(mode==Mode.GRU):
    updrsPerPatient_train,updrsPerPatient_train_labels, updrsPerPatient_Test,updrsPerPatient_Test_labels = loadInstance.GetUpdrsPerPatient(updrs_visits)
    gru_model = dlNetwork.build_GRU_network(len(updrsPerPatient_train[0]))
    gru_history = gru_model.fit(updrsPerPatient_train,updrsPerPatient_train_labels,epochs=205, validation_split=0.2, verbose=1)
    dlNetwork.plotLoss(gru_history)
    gru_model.predict(updrsPerPatient_Test)
    updrsPerPatient_Test_labels

###################################################################################
##  This network Predict updrs for all patian visits by peptide abundance vector data
##################################################################################

if(mode==Mode.FULLY_CONNECTED):
    peptide_train, peptide_test, updrs_train, updrs_test =dlNetwork.GetTrainAndTestSets()
    Fc_model = dlNetwork.buildFullyConnectedNetwork()
    fc_history = Fc_model.fit(peptide_train, updrs_train, epochs=85, validation_split=0.2, verbose=1)
    dlNetwork.plotLoss(fc_history)
    Fc_model.predict(peptide_test)
    updrs_test
#################################################################################################
##  This network Predict updrs for Visit_60 both by priod progression of updrs and peptide abundance
################################################################################################

if(mode==Mode.MULTI_INPUT_LSTM_LSTM):
    print ("zevel")


#model = Model(inputs=[input1, input2], outputs=output)
#
## Compile the model
#model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])
#
## Fit the model to the data
#model.fit([X1, X2], y, epochs=10, batch_size=32)



# prediction model design 

#lstm per each patient- predict udprs in visit 36
#create dataset to lstm based on udprs_visits_vector , updrs_visits
#data set composed of updrs per patient all visits where need to predict last visit
# 55_0(u1 u2 u3 u4) 
# 55_3(u1 u2 u3 u4)
# 55_6(u1 u2 u3 u4)
# 55_9(u1 u2 u3 u4)
#...
# predict  55_36(u1 u2 u3 u4)
# 20 % of patients are test cases 80 % of patients for training


# patients_data array which compose of
# fit(patient_data) -> [Patient_ID, Visit_month, peptide_type,peptideAbundance ]
#input_data(visit_month (0-~36),)
#target data udprs per month 0,6,12,24
# ~160 to ~210 peptides types taken per visit 
#lstm that get timebased visit peptides e.g : 
#                V1  V2  V3  V4
#  visit month    0   6    12  24   36  48  60  108
#  Pep1           21   45  42  97               .
#  Pep2           2    32  44  111              .
#  Pep3           15   51  511 511              .
#  .
#  .
#  .
#  Protein i Val  21   545  65  90
# udprs_1          3    5   8   9
# udprs_2          12  14   13   15            
# udprs_3          22  34   45   34
# udprs_4          22  45   56   45
# targets are udprs[i] - for fit #1  udprs month #6  for fit2 udprs for M12  
#as first step 
# we got solid data for months 0,6,12,24,36,48,60,108
# train on data up to M48 predict udprs for M60 

#for step 2  - layer concatenation
#  udprs history
#  peptide history
# concatenate into decision
#
#from tensorflow.keras.layers import Input, Dense, concatenate
#from tensorflow.keras.models import Model
#
## Define the input layers
#input1 = Input(shape=(100,))
#input2 = Input(shape=(50,))
#
## Define the first hidden layer for input 1
#hidden1 = Dense(64, activation='relu')(input1)
#
## Define the second hidden layer for input 2
#hidden2 = Dense(32, activation='relu')(input2)
## Define the first GRU layer for input 1
#gru1 = GRU(32)(input1)
#
## Define the second GRU layer for input 2
#gru2 = GRU(16)(input2)
## Combine the two inputs and their hidden layers
#concatenated = concatenate([hidden1, hidden2 ,gru1,gru2])
#
## Define the output layer
#output = Dense(1, activation='sigmoid')(concatenated)
#
## Define the model
#model = Model(inputs=[input1, input2], outputs=output)
#
## Compile the model
#model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
#
## Fit the model to the data
#model.fit([X1, X2], y, epochs=10, batch_size=32)


##### first input to keras API  - 
### predict  one hoe encoding for peptide data  to current  udprs values
# 1. custom_encoded layout (one hot encoding+ value per each) for normalized peptide + uniprot per visit - regardles for number of visit and patient id
# 2. input to fully connected network and predict udprs1-4
# 3.  

##### third input to keras API  - 
### predict  one hoe encoding for uniprot data  to current  udprs values
# 1. custom_encoded layout (one hot encoding+ value per each) for normal

# import numpy as np
# from keras.utils import to_categorical

# # Sample list of categorical labels
# labels = ['cat', 'dog', 'bird', 'dog', 'cat']

# Define a custom mapping of labels to custom values
# custom_encoding = {'cat': 2, 'dog': 5, 'bird': 7}

# # Use a list comprehension to map labels to custom values
# custom_encoded_labels = [custom_encoding[label] for label in labels]

# # Optionally, convert the custom encoded labels to one-hot encoding
# one_hot_encoded_labels = to_categorical(custom_encoded_labels)

# print(one_hot_encoded_labels)

# model = models.Sequential()
# model.add(layers.Dense(64, activation='relu', input_shape=(10000,)))
# model.add(layers.Dense(64, activation='relu'))
# model.add(layers.Dense(46, activation='softmax'))

# model.compile(optimizer='rmsprop',
#               loss='categorical_crossentropy',
#               metrics=['accuracy'])

# x_val = x_train[:1000]
# partial_x_train = x_train[1000:]

# y_val = one_hot_train_labels[:1000]
# partial_y_train = one_hot_train_labels[1000:]

# history = model.fit(partial_x_train,
#                     partial_y_train,
#                     epochs=20,
#                     batch_size=512,
#                     validation_data=(x_val, y_val))


##second input 
# lstm network predict udprs for target monthes (24,36) by given udprs