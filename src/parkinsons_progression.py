from loadAndPreprocess import LoadAndPreprocess



protein_train_path = "..\\amp-parkinsons-disease-progression-prediction\\train_proteins.csv"
peptide_train_path = "..\\amp-parkinsons-disease-progression-prediction\\train_peptides.csv"
train_clinical_data ="..\\amp-parkinsons-disease-progression-prediction\\train_clinical_data.csv"

loadInstance = LoadAndPreprocess(protein_train_path, peptide_train_path, train_clinical_data)

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
