


#load data from 
# https://www.kaggle.com/c/amp-parkinsons-disease-progression-prediction

import shutil,os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from keras.utils import to_categorical

class LoadAndPreprocess:
    
    
    def __init__(self,protein_train_path,peptide_train_path,clinical_data_path):
        self.protein_train_path = protein_train_path
        self.peptide_train_path = peptide_train_path
        self.clinical_data_path = clinical_data_path
        #self.GetPeptideData()
        #def GetPeptideDataPerVisit()
        self.visit_id = 0
        self.peptideList = 0
        self.peptide_visits_vector =  0 # np.empty((0,int(BATCH_LEN/ONLY_POSITIVE_FREQS)))
        self.udprs_vistis_vector = 0

    def GetUdprsData(self):
        df = pd.read_csv(self.clinical_data_path)
        df.fillna(0, inplace=True)
        unique_visits = df['visit_id'].unique()
        self.udprs_vistis_vector = np.empty((0,4))
        for uniq_visit in unique_visits:
            visit_data = df[df['visit_id']== uniq_visit] 
            udprs_1 = visit_data['updrs_1'].tolist()[0]
            udprs_2 = visit_data['updrs_2'].tolist()[0]
            udprs_3 = visit_data['updrs_3'].tolist()[0]
            udprs_4 = visit_data['updrs_4'].tolist()[0]
            updrs = [udprs_1,udprs_2,udprs_3,udprs_4]
            self.udprs_vistis_vector = np.vstack((self.udprs_vistis_vector , updrs))
        return self.udprs_vistis_vector , unique_visits

    def GetPeptideData(self):
        df = pd.read_csv(self.peptide_train_path)
        #pd.Dataframe(df)
        unique_visits = df['visit_id'].unique()
        self.peptideList = df['Peptide'].unique()
        self.peptide_visits_vector = np.empty((0,len(self.peptideList)))
        #create fixed LUT for peptides. then fill with values per visit
        for uniq_visit in unique_visits:
            visit_data = df[df['visit_id']== uniq_visit] 
            visit_peptide_vector = self.GetPeptideDataPerVisit(visit_data)  
            self.peptide_visits_vector = np.vstack((self.peptide_visits_vector , visit_peptide_vector))
        
        return self.peptide_visits_vector , unique_visits  

    #this function or another should also get udprs vals in adequate to peptides values
    def GetPeptideDataPerVisit(self,visit_data):
        visit_dict = dict(zip(visit_data['Peptide'], visit_data['PeptideAbundance']))
        peptide_vector = np.zeros(len(self.peptideList))
        for peptide , value in visit_dict.items():
            #fill in array 0 for NA in visit_data or value 
            peptide_idx = np.where(self.peptideList == peptide)
            peptide_vector[peptide_idx] = value
            #print (peptide)
        return peptide_vector