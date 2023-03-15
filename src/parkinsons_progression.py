from loadAndPreprocess import LoadAndPreprocess



protein_train_path = "..\\amp-parkinsons-disease-progression-prediction\\train_proteins.csv"
peptide_train_path = "..\\amp-parkinsons-disease-progression-prediction\\train_peptides.csv"
train_clinical_data ="..\\amp-parkinsons-disease-progression-prediction\\train_clinical_data.csv"

loadInstance = LoadAndPreprocess(protein_train_path, peptide_train_path, train_clinical_data)

# prediction model design 

#lstm network recieve as input 
# patients_data array which compose of
# fit(patient_data) -> [Patient_ID, Visit_month, peptide_type,peptideAbundance ]
#input_data(visit_month (0-~36),)
#target data udprs per month 0,6,12,24
# ~160 to ~210 peptides types taken per visit 
#lstm that get timebased visit peptides e.g : 
#                V1  V2  V3  V4
#  visit month    0   3    6   12
#  Pep1           21   45  42  97
#  Pep2           2    32  44  111 
#  Pep3           15   51  511  511
#  PRotein i Val 21   545  65  90
# targets are udprs  