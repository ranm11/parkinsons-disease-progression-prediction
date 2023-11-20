from loadAndPreprocess import LoadAndPreprocess
from DataProccessing import DLNetwork


protein_train_path = "parkinsons-disease-progression-prediction\\amp-parkinsons-disease-progression-prediction\\train_proteins.csv"
peptide_train_path = "parkinsons-disease-progression-prediction\\amp-parkinsons-disease-progression-prediction\\train_peptides.csv"
train_clinical_data ="parkinsons-disease-progression-prediction\\amp-parkinsons-disease-progression-prediction\\train_clinical_data.csv"

loadInstance = LoadAndPreprocess(protein_train_path, peptide_train_path, train_clinical_data)
udprs_visits_vector , updrs_visits = loadInstance.GetUdprsData()
peptide_visits_vector , peptide_visits = loadInstance.GetPeptideData()
#create dictionaries
peptide_dict = dict(zip(peptide_visits, peptide_visits_vector))
udprs_dict = dict(zip(updrs_visits, udprs_visits_vector))
common_visits = list(set(updrs_visits).intersection(peptide_visits))

print("zevel")  
dlNetwork = DLNetwork(common_visits,udprs_dict,peptide_dict)
peptide_train, peptide_test, updrs_train, updrs_test =dlNetwork.GetTrainAndTestSets()
Fc_model = dlNetwork.buildFullyConnectedNetwork()
history = Fc_model.fit(peptide_train, updrs_train, epochs=85, validation_split=0.2, verbose=1)
# prediction model design 


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