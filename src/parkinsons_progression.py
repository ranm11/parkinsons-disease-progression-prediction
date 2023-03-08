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