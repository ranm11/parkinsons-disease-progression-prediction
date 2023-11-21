
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM, GRU , Embedding, Conv1D , MaxPooling1D
from tensorflow.keras.layers import Dropout
from tensorflow.keras import Input , Model
import numpy as np

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

    def GetTrainAndTestSets(self):
        self.peptideList = np.empty((0,self.input_len))
        self.updrsList = np.empty((0,4))
        for key in self.peptide_dict:
            if key in self.peptide_dict and key in self.udprs_dict:
                self.peptideList = np.vstack((self.peptideList,self.peptide_dict[key]))
                self.updrsList = np.vstack((self.updrsList,self.udprs_dict[key]))
        return self.peptideList[:self.trainSetNumber] , self.peptideList[self.trainSetNumber:],self.updrsList[:self.trainSetNumber],self.updrsList[self.trainSetNumber:]    
    
    def DataNormalization(self):
        print("datanormalization")

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