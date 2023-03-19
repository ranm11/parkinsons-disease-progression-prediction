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
#  visit month    0   6    12  24   36  48  60  108
#  Pep1           21   45  42  97               .
#  Pep2           2    32  44  111              .
#  Pep3           15   51  511  511             .
#  .
#  .
#  .
#  Protein i Val  21   545  65  90
# udprs_1          3    5   (8)   9
# udprs_2          12  14  (13)   15            
# udprs_3          22  34  (45)   34
# udprs_4          22  45  (56)   45
# targets are udprs[i] - for fit #1  udprs month #6  for fit2 udprs for M12  
#predict values in parentisis (val) for monthes 6 12 24 