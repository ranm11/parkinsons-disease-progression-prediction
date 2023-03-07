


#load data from 
# https://www.kaggle.com/c/amp-parkinsons-disease-progression-prediction

import shutil,os
import matplotlib.pyplot as plt
import numpy as np

class LoadAndPreprocess:
    def __init__(self,protein_train_path,peptide_train_path,clinical_data_path):
        self.protein_train_path = protein_train_path
        self.peptide_train_path = peptide_train_path
        self.clinical_data_path = clinical_data_path

    def GetPeptideData(self):
        