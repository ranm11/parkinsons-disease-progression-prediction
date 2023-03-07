from loadAndPreprocess import LoadAndPreprocess



protein_train_path = "..\\amp-parkinsons-disease-progression-prediction\\train_proteins.csv"
peptide_train_path = "..\\amp-parkinsons-disease-progression-prediction\\train_peptides.csv"
train_clinical_data ="..\\amp-parkinsons-disease-progression-prediction\\train_clinical_data.csv"

loadInstance = LoadAndPreprocess(protein_train_path, peptide_train_path, train_clinical_data)
