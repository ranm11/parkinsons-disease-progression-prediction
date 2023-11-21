
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM, GRU , Embedding, Conv1D , MaxPooling1D
from tensorflow.keras.layers import Dropout
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

    def GetTrainAndTestSets(self):
        self.peptideList = np.empty((0,self.input_len))
        self.updrsList = np.empty((0,4))
        for key in self.peptide_dict:
            if key in self.peptide_dict and key in self.udprs_dict:
                self.peptideList = np.vstack((self.peptideList,self.peptide_dict[key]))
                self.updrsList = np.vstack((self.updrsList,self.udprs_dict[key]))
        self.DataNormalization()
        return self.peptideListNormalized[:self.trainSetNumber] , self.peptideListNormalized[self.trainSetNumber:],self.updrsList[:self.trainSetNumber],self.updrsList[self.trainSetNumber:]    
    
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

        #plt.subplot(212)
        #acc = history.history['accuracy']
        #plt.plot(epochs, acc, 'bo', label='Training acc')
        #plt.title('Training and validation accuracy')
        #plt.xlabel('Epochs')
        #plt.ylabel('Loss')
        #plt.legend()
        #plt.show()
