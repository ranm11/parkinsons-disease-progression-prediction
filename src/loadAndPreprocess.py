


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
        self.uniq_patient = 0
        self.Common_Visit_Number_in_Patients = 100
        self.final_patient_records_list = 0
        self.updrs_train_set = 70

    def GetPatientSubset(self,most_Common_visits_indexes,patient_record ):
        patiend_id = [x.rsplit('_',2)[0] for x in patient_record][0]
        patient_subset = [f"{patiend_id}_{value}" for  value in most_Common_visits_indexes]
        return patient_subset
    
    def GetUpdrsPerPatient(self,unique_visits):
        self.udprs_vistis_vector_lists = np.empty((0,12,4))
        self.final_patient_records_list = np.empty((0,12))
        visitToUdprs_dict = dict(zip(unique_visits, self.udprs_vistis_vector))
        list_of_lists = [[] for _ in range(len(self.uniq_patient))]
        index = 0
        for uniq_patient in self.uniq_patient:
            patient_visits = np.array([x for x in unique_visits if x.startswith(uniq_patient)])
            list_of_lists[index] = patient_visits
            index=index+1
            #set into list of lists
        filtered_lists = [sublist for sublist in list_of_lists if len(sublist) > 9]  # remove small visits number patients
        flattened_list = [item for sublist in filtered_lists for item in sublist]
        flattened_list_nums = [x.rsplit('_',2)[1] for x in flattened_list]
        visit_min_val = min(flattened_list_nums)
        visit_max_val = max(flattened_list_nums)
        histogram = np.array([flattened_list_nums.count(str(i)) for i in range(int(visit_min_val), int(visit_max_val) + 1)])
        most_Common_visits_indexes = np.where(histogram > self.Common_Visit_Number_in_Patients)[0]
        # return onlt arrays from flattened_list which are subsets of most_Common_visits_indexes
        #subarrays = [arr for arr in np.array(filtered_lists) if np.isin(arr, most_Common_visits_indexes).all()]
        for patient_record in filtered_lists:
            patient_visits_record = [x.rsplit('_',2)[1] for x in patient_record]
            if np.isin(most_Common_visits_indexes.astype(str),np.array(patient_visits_record)).all():
                patient_subset = self.GetPatientSubset(most_Common_visits_indexes.astype(str),patient_record)
                self.final_patient_records_list=np.vstack((self.final_patient_records_list,patient_subset))  
                updrs_for_patient = [visitToUdprs_dict.get(key) for key in patient_subset] 
                contains_none = any(element is None for element in updrs_for_patient)
                if contains_none == True:
                    continue
                self.udprs_vistis_vector_lists  = np.vstack((self.udprs_vistis_vector_lists,np.array(updrs_for_patient)[np.newaxis,:,:]))
        self.UpdrsDataNormalization()
        return self.udprs_vistis_vector_lists[:self.updrs_train_set,:-1,:], self.udprs_vistis_vector_lists[:self.updrs_train_set,-1:,:] , self.udprs_vistis_vector_lists[self.updrs_train_set:,:-1,:] ,self.udprs_vistis_vector_lists[self.updrs_train_set:,-1:,:]
        #listOfVisitPerPatient = 

    def UpdrsDataNormalization(self):
        #mean = (self.udprs_vistis_vector_lists).mean(axis=0)
        self.udprs_vistis_vector_lists = self.udprs_vistis_vector_lists / 25
        # std = Normalized_Data.std(axis=0)
        # self.udprs_vistis_vector_lists = Normalized_Data/std

    def GetUdprsData(self):
        df = pd.read_csv(self.clinical_data_path)
        df.fillna(0, inplace=True)
        unique_visits = df['visit_id'].unique()
        self.udprs_vistis_vector = np.empty((0,4))
        self.uniq_patient = np.empty((0,1))
        for uniq_visit in unique_visits:
            visit_data = df[df['visit_id']== uniq_visit] 
            udprs_1 = visit_data['updrs_1'].tolist()[0]
            udprs_2 = visit_data['updrs_2'].tolist()[0]
            udprs_3 = visit_data['updrs_3'].tolist()[0]
            udprs_4 = visit_data['updrs_4'].tolist()[0]
            updrs = [udprs_1,udprs_2,udprs_3,udprs_4]
            self.udprs_vistis_vector = np.vstack((self.udprs_vistis_vector , updrs))
            self.uniq_patient = np.vstack((self.uniq_patient , uniq_visit.rsplit('_',2)[0]))
        self.uniq_patient =  np.unique(self.uniq_patient)
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