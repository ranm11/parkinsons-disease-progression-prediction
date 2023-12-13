
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM, GRU , Embedding, Conv1D , MaxPooling1D
from tensorflow.keras.layers import Dropout , concatenate
from tensorflow.keras import Input , Model
import numpy as np
import matplotlib.pyplot as plt

class DLNetwork:
    def __init__(self,common_visits,udprs_dict,peptide_dict):
        #Visit Data Clas Array sorted by visir month
        self.common_visits=common_visits
        self.udprs_dict=udprs_dict
        self.peptide_dict=peptide_dict
        self.input_len = len(peptide_dict[common_visits[0]])    #peptide input vector len
        self.trainSetNumber = 800
        self.peptideList = 0
        self.updrsList = 0
        self.peptideListNormalized = 0

    ############################################################
    #this function return peptide list and its coresponding updrs list    
    ############################################################
    def GetTrainAndTestSets(self):
        self.peptideList = np.empty((0,self.input_len))
        self.updrsList = np.empty((0,4))
        for key in self.peptide_dict:
            if key in self.peptide_dict and key in self.udprs_dict:
                self.peptideList = np.vstack((self.peptideList,self.peptide_dict[key]))
                self.updrsList = np.vstack((self.updrsList,self.udprs_dict[key]))
        self.DataNormalization()
        return self.peptideListNormalized[:self.trainSetNumber] , self.peptideListNormalized[self.trainSetNumber:],self.updrsList[:self.trainSetNumber]/25,self.updrsList[self.trainSetNumber:]/25    
    
    def DataNormalization(self):
        print("datanormalization")
        mean = (self.peptideList).mean(axis=0)
        Normalized_train_Data = self.peptideList - mean
        std = Normalized_train_Data.std(axis=0)
        self.peptideListNormalized = Normalized_train_Data/std

    def buildFullyConnectedNetwork(self):
        inputs = Input(shape=(self.input_len,),name="peptides_inputs")
        layer1 = Dense(units = 800,activation = 'relu')(inputs) 
        layer1 = Dropout(0.1)(layer1)
        layer1 = Dense(units=500, activation='relu')(layer1)
        layer1 = Dropout(0.1)(layer1)
        layer1 = Dense(units  = 128, activation='sigmoid')(layer1)
        layer1 = Dropout(0.1)(layer1)
        output= Dense(4,activation='linear')(layer1)
        model = Model(inputs,output)
        model.compile(optimizer='adam',  loss='mean_squared_error', metrics=['mae'])
        return model
    
    def plotLoss(self,history):
        plt.figure()
        plt.subplot(211)
        loss = history.history['loss']
        epochs = range(1, len(loss) + 1)
        plt.plot(epochs, loss, 'bo', label='Training loss')
        plt.title('Training and validation loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.show()

    def build_GRU_network(self,updrsInputLen):
        input = Input(shape=(updrsInputLen,4))
        gru1 = GRU(20)(input)
        out_layer = Dense(4,activation='linear')(gru1)
        model = Model(inputs=input,outputs= out_layer)
        model.compile(loss='mean_squared_error', optimizer= 'adam', metrics=['mae'])
        return model

    def build_multi_input_network(self,updrsInputLen):
        gru_input = Input(shape=(updrsInputLen,4))
        gru_out_layer = GRU(20)(gru_input)

        fc_input = Input(shape=(self.input_len,),name="peptides_inputs")
        layer1 = Dense(units = 800,activation = 'relu')(fc_input) 
        layer1 = Dropout(0.1)(layer1)
        layer1 = Dense(units=500, activation='relu')(layer1)
        layer1 = Dropout(0.1)(layer1)
        fc_output_layer = Dense(units  = 128, activation='sigmoid')(layer1)

        #layers concatenation
        concatenated = concatenate([gru_out_layer, fc_output_layer]) 

        # Output layer
        #output = Dense(4, activation='linear')(concatenated)
        output = Dense(4, activation='linear')(concatenated)
        model = Model(inputs=[gru_input, fc_input], outputs=output)
        model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])
        return model
    
    def build_3_input_network(self,updrsInputLen,proteinInputLen):
        #updrs GRU
        UPDRS_LEN = 4
        PROTEIN_VEC_LEN = 227
        gru_updrs_input = Input(shape=(updrsInputLen,4))
        gru_updrs_out_layer = GRU(20)(gru_updrs_input)
        #protein GRU
        gru_protein_input = Input(shape=(proteinInputLen,227))
        gru_protein_out_layer = GRU(20)(gru_protein_input)
        #FC - NN
        fc_input = Input(shape=(self.input_len,),name="peptides_inputs")
        layer1 = Dense(units = 800,activation = 'relu')(fc_input) 
        layer1 = Dropout(0.1)(layer1)
        layer1 = Dense(units=500, activation='relu')(layer1)
        layer1 = Dropout(0.1)(layer1)
        fc_output_layer = Dense(units  = 128, activation='sigmoid')(layer1)

        #layers concatenation
        concatenated = concatenate([gru_updrs_out_layer,gru_protein_out_layer, fc_output_layer]) 

        # Output layer
        #output = Dense(4, activation='linear')(concatenated)
        output = Dense(4, activation='linear')(concatenated)
        model = Model(inputs=[gru_updrs_input,gru_protein_input, fc_input], outputs=output)
        model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])
        return model