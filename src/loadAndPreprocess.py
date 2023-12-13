


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
        self.updrs_unique_visits = 0
        self.Min_NOF_Patients_required = 120  #100
        self.final_patient_records_list = 0
        self.updrs_train_set = 65  #28
        self.visit_updrs_dict = {}
        #peptides dataset
        self.peptide_vector_for_last_visit=0
        self.peptide_dict=0
        self.peptideListPerPatient = 0
        self.protein_MIN_NOF_of_Patients = 120
        #proteinDataset
        self.protein_per_patient_table=0
        self.PatientsProteinTrainingSetLen = 75

    def GetPeptidesVisitsHistogram(self):
        peptide_keys_list = list(self.peptide_dict.keys())
        #peptide_keys_list_0 = np.array([x for x in peptide_keys_list if x.endswith('_0')])
        flattened_list_nums = [x.rsplit('_',2)[1] for x in peptide_keys_list]
        visit_min_val = min(flattened_list_nums)
        visit_max_val = max(flattened_list_nums)
        histogram = np.array([flattened_list_nums.count(str(i)) for i in range(int(visit_min_val), int(visit_max_val) + 1)])
        return histogram

    def GetPatientSubset(self,most_Common_visits_indexes,patient_record ):
        patiend_id = [x.rsplit('_',2)[0] for x in patient_record][0]
        patient_subset = [f"{patiend_id}_{value}" for  value in most_Common_visits_indexes]
        return patient_subset
    
    # the thought was to add lstm network for peptides - but there is no long enough patient sequencr for this 
    def GetPeptidePerVisitsSubset(self,patient_visit_subset,peptide_histogram):
        most_Common_visits_indexes = np.where(peptide_histogram > self.Min_NOF_Patients_required)[0]
        #verify keys specified in most_Common_visits_indexes exists in self.peptide_dict for current patient
        most_Common_visits_indexes_str = [str(value) for value in most_Common_visits_indexes]
        most_Common_visits_indexes_str = ['_0', '_24', '_48']
        relevant_keys_for_currect_patient  = [x for x in patient_visit_subset if any(x.endswith(end) for end in most_Common_visits_indexes_str)]
        all_keys_exist = all(key in self.peptide_dict for key in relevant_keys_for_currect_patient)
        if all_keys_exist:
            prptide_list_per_patient = [self.peptide_dict[key] for key in relevant_keys_for_currect_patient if key in self.peptide_dict]
            self.peptideListPerPatient = np.vstack((self.peptideListPerPatient,np.array(prptide_list_per_patient)[np.newaxis,:,:]))

    def GetUpdrsPerPatient(self,unique_visits):
        NOF_VISITS = 9   #12 
        NOF_PATIENTS = 70
        self.udprs_vistis_vector_lists = np.empty((0,NOF_VISITS,4))
        self.peptide_vector_for_last_visit = np.empty((0,968))
        self.final_patient_records_list = np.empty((0,NOF_VISITS))
        self.peptideListPerPatient = np.empty((0,3,968))
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
        peptide_histogram = self.GetPeptidesVisitsHistogram()
        most_Common_visits_indexes = np.where(histogram > self.Min_NOF_Patients_required)[0]
        # return onlt arrays from flattened_list which are subsets of most_Common_visits_indexes
        #subarrays = [arr for arr in np.array(filtered_lists) if np.isin(arr, most_Common_visits_indexes).all()]
        for patient_record in filtered_lists:
            patient_visits_record = [x.rsplit('_',2)[1] for x in patient_record]
            if np.isin(most_Common_visits_indexes.astype(str),np.array(patient_visits_record)).all():
                patient_subset = self.GetPatientSubset(most_Common_visits_indexes.astype(str),patient_record)
                self.final_patient_records_list=np.vstack((self.final_patient_records_list,patient_subset))  
                updrs_for_patient = [visitToUdprs_dict.get(key) for key in patient_subset] 
                contains_none = any(element is None for element in updrs_for_patient)
                if contains_none == True or patient_subset[-1] not in self.peptide_dict:
                    continue
                self.udprs_vistis_vector_lists  = np.vstack((self.udprs_vistis_vector_lists,np.array(updrs_for_patient)[np.newaxis,:,:]))
                self.peptide_vector_for_last_visit = np.vstack((self.peptide_vector_for_last_visit,self.peptide_dict[patient_subset[-1]]))
                #self.GetPeptidePerVisitsSubset(patient_subset,peptide_histogram)
        self.UpdrsDataNormalization()
        #visit_subset is (0,6,12,24,36,48)
        return self.udprs_vistis_vector_lists[:self.updrs_train_set,:-1,:], self.udprs_vistis_vector_lists[:self.updrs_train_set,-1:,:] , self.udprs_vistis_vector_lists[self.updrs_train_set:,:-1,:] ,self.udprs_vistis_vector_lists[self.updrs_train_set:,-1:,:]

    def GetpeptidePerLastVisit(self):
        self.PeptideDataNormalization()
        return self.peptide_vector_for_last_visit[:self.updrs_train_set],self.peptide_vector_for_last_visit[self.updrs_train_set:]
    
    def PeptideDataNormalization(self):
        mean = (self.peptide_vector_for_last_visit).mean(axis=0)
        Normalized_train_Data = self.peptide_vector_for_last_visit - mean
        std = Normalized_train_Data.std(axis=0)
        self.peptide_vector_for_last_visit = Normalized_train_Data/std

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
        self.updrs_unique_visits = np.empty((0,1))
        for uniq_visit in unique_visits:
            visit_data = df[df['visit_id']== uniq_visit] 
            udprs_1 = visit_data['updrs_1'].tolist()[0]
            udprs_2 = visit_data['updrs_2'].tolist()[0]
            udprs_3 = visit_data['updrs_3'].tolist()[0]
            udprs_4 = visit_data['updrs_4'].tolist()[0]
            updrs = [udprs_1,udprs_2,udprs_3,udprs_4]
            self.udprs_vistis_vector = np.vstack((self.udprs_vistis_vector , updrs))
            self.uniq_patient = np.vstack((self.uniq_patient , uniq_visit.rsplit('_',2)[0]))
            self.updrs_unique_visits = np.vstack((self.updrs_unique_visits , uniq_visit))
            self.visit_updrs_dict[uniq_visit] = updrs
        self.uniq_patient =  np.unique(self.uniq_patient)
        return self.udprs_vistis_vector , self.updrs_unique_visits[:,0] , self.visit_updrs_dict

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
        
        self.peptide_dict = dict(zip(unique_visits, self.peptide_visits_vector))
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
    
    def GetMultiInputDataSets(self,peptide_dict,udprs_dict,common_visits):
        self.peptideList_for_multi_input = np.empty((0,len(peptide_dict[common_visits[0]])))
        self.updrsList = np.empty((0,4))
        for key in peptide_dict:
            if key in peptide_dict and key in udprs_dict:
                self.peptideList = np.vstack((self.peptideList,self.peptide_dict[key]))
                self.updrsList = np.vstack((self.updrsList,self.udprs_dict[key]))
        self.DataNormalization()
        return self.peptideListNormalized[:self.trainSetNumber] , self.peptideListNormalized[self.trainSetNumber:],self.updrsList[:self.trainSetNumber],self.updrsList[self.trainSetNumber:]    
    
    def ParsePatientBlk(self,uniprot_to_value_dict_patient_visit_i, uniq_uniport_list):
        protein_vector_for_patient_visit= np.zeros(len(uniq_uniport_list))
        for  Protein,value in uniprot_to_value_dict_patient_visit_i.items():
            #fill in array 0 for NA in visit_data or value 
            protein_idx = np.where(uniq_uniport_list == Protein)
            protein_vector_for_patient_visit[protein_idx] = value
            #print (peptide)
        return protein_vector_for_patient_visit
    
    def ProteinNormalization(self):
        mean = (self.protein_per_patient_table).mean(axis=0)
        Normalized_train_Data = self.protein_per_patient_table - mean
        std = Normalized_train_Data.std(axis=0)
        self.protein_per_patient_table = Normalized_train_Data/std


    def GetProteinInputDataset(self):
        df = pd.read_csv(self.protein_train_path)
        #df.fillna(0, inplace=True)
        #unique_visits = df['UniProt'].unique()
        unique_visits = df['visit_id'].unique()
        visits_ids_for_unique_visits = [x.rsplit('_',2)[1] for x in unique_visits]
        visit_min_val = min(visits_ids_for_unique_visits)
        visit_max_val = max(visits_ids_for_unique_visits)
        protein_visits_histogram = np.array([visits_ids_for_unique_visits.count(str(i)) for i in range(int(visit_min_val), int(visit_max_val) + 1)])
        self.most_Common_visits_indexes_protein = np.where(protein_visits_histogram > self.protein_MIN_NOF_of_Patients)[0]
        most_Common_visits_indexes_protein = np.array([0,12,48])
        uniq_uniport_list = df['UniProt'].unique()
        unique_patients = df['patient_id'].unique()
        protein_per_patient_table = np.empty((0,len(most_Common_visits_indexes_protein),len(uniq_uniport_list)))
        visit_protein_dict = {}
        for unique_patient in unique_patients:
            patientBlk = df[df['patient_id']== unique_patient]
            patient_visits = patientBlk['visit_month'].unique()    
            if not set(most_Common_visits_indexes_protein).issubset(set(patient_visits)):
                continue
            protein_per_visit_table = np.empty((0,len(uniq_uniport_list)))
            visit_id_vector = np.empty((0,1))
            for  visit in most_Common_visits_indexes_protein:
                visit_i = patientBlk[patientBlk['visit_month'] == visit ]
                NPX = visit_i['NPX']
                UniProt = visit_i['UniProt']
                visit_id = visit_i['visit_id'].unique()
                visit_id_vector = np.vstack((visit_id_vector,visit_id))
                uniprot_to_value_dict_patient_visit_i = dict(zip(UniProt, NPX))
                protein_vector_for_patient_visit = self.ParsePatientBlk(uniprot_to_value_dict_patient_visit_i, uniq_uniport_list)
                protein_per_visit_table = np.vstack((protein_per_visit_table,protein_vector_for_patient_visit)) # redundant
            
            #check if   visit_vector_id has record 48
            if  set(visit_id).issubset(set(np.array(self.updrs_unique_visits[:,0]))):
                visit_protein_dict[visit_id[0]] = protein_per_visit_table
                

            #protein_per_patient_table = np.vstack((protein_per_patient_table, np.array(protein_per_visit_table)[np.newaxis,:,:]))    #redundant
            
        #self.protein_per_patient_table=protein_per_patient_table
        #self.ProteinNormalization()
        return   visit_protein_dict
            #self.peptide_visits_vector = np.vstack((self.peptide_visits_vector , visit_peptide_vector))

    #this function compose dataset of updrs_48 
    #go over visit_protein_dict - foreach key check if key exist in self.visit_updrs_dict
    # if yes - make list of all  9 visits for this patient correspomding to visit_protein_dist key 
    def GetProteinDatasets(self,visit_protein_dict):
        NOF_VISITS = 6   
        TRAIN_DATASET_LEN = 53
        TEST_DATASET_LEN = 4
        udprs_vistis_vector_lists = np.empty((0,NOF_VISITS,4))
        protein_visit_vector = np.empty((0,3,227))
        peptide_list = np.empty((0,968))
        self.most_Common_visits_indexes_protein
        for visit,proteins in visit_protein_dict.items():
            if visit in self.visit_updrs_dict:
                #keys_for_patients = {key for key in self.visit_updrs_dict.keys() if key.startswith(visit.rsplit('_',2)[0])} 
                keys_for_patients = [visit.rsplit('_',2)[0] + '_' + str(x) for x in self.most_Common_visits_indexes_protein]
                all_keys_exist = all(key in self.visit_updrs_dict for key in keys_for_patients)
                if all_keys_exist:
                    updrs_for_patient = [self.visit_updrs_dict.get(key) for key in keys_for_patients]
                    udprs_vistis_vector_lists = np.vstack((udprs_vistis_vector_lists,np.array(updrs_for_patient)[np.newaxis,:,:]))
                    protein_visit_vector=np.vstack((protein_visit_vector,np.array(visit_protein_dict[visit])[np.newaxis,:,:]))
                    peptide_list = np.vstack((peptide_list,self.peptide_dict[visit]))
        peptide_list= self.Normalize(peptide_list)
        protein_visit_vector = self.Normalize(protein_visit_vector)
        udprs_vistis_vector_lists = udprs_vistis_vector_lists/25
        #updrsPerPatient_train, updrsPerPatient_train_labels,updrsPerPatient_test,updrsPerPatient_Test_labels, proteins_train,proteins_test, peptides_train, peptides_test,  ,                    
        return udprs_vistis_vector_lists[:TRAIN_DATASET_LEN,:NOF_VISITS-1] ,udprs_vistis_vector_lists[:TRAIN_DATASET_LEN,-1:,:],udprs_vistis_vector_lists[-TEST_DATASET_LEN:,:NOF_VISITS-1],udprs_vistis_vector_lists[-TEST_DATASET_LEN:,-1] ,protein_visit_vector[:TRAIN_DATASET_LEN] ,protein_visit_vector[-TEST_DATASET_LEN:], peptide_list[:TRAIN_DATASET_LEN] , peptide_list[-TEST_DATASET_LEN:]
        
    def Normalize(self,data):
        mean = data.mean(axis=0)
        Normalized_train_Data = data - mean
        std = Normalized_train_Data.std(axis=0)
        return Normalized_train_Data/std

