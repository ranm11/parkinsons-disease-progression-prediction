from loadAndPreprocess import LoadAndPreprocess
from DataProccessing import DLNetwork
from enum import Enum
from keras.utils import plot_model

protein_train_path = "parkinsons-disease-progression-prediction\\amp-parkinsons-disease-progression-prediction\\train_proteins.csv"
peptide_train_path = "parkinsons-disease-progression-prediction\\amp-parkinsons-disease-progression-prediction\\train_peptides.csv"
train_clinical_data ="parkinsons-disease-progression-prediction\\amp-parkinsons-disease-progression-prediction\\train_clinical_data.csv"

class Mode(Enum):
    GRU = 1
    FULLY_CONNECTED = 2
    TWO_INPUT_FC_LSTM = 3
    MULTI_INPUT_LSTM = 4


loadInstance = LoadAndPreprocess(protein_train_path, peptide_train_path, train_clinical_data)
updrs_visits = loadInstance.GetUdprsData()
visit_protein_dict = loadInstance.GetProteinInputDataset()
peptide_visits_vector , peptide_visits ,peptide_dict = loadInstance.GetPeptideData()

#create dictionaries
common_visits = list(set(updrs_visits).intersection(peptide_visits))

dlNetwork = DLNetwork(len(peptide_dict[common_visits[0]]))


mode = Mode.MULTI_INPUT_LSTM

#################################################################################################
##  This network Predict updrs for Visit_48 both by priod progression of updrs protein of 3 visits and peptide abundance
################################################################################################

if(mode==Mode.MULTI_INPUT_LSTM):
    updrsPerPatient_train, updrsPerPatient_train_labels, updrsPerPatient_Test, updrsPerPatient_Test_labels, proteins_train, proteins_test, peptides_train, peptides_test = loadInstance.GetProteinDatasets(visit_protein_dict)
    multi_input_model = dlNetwork.build_3_input_network(len(updrsPerPatient_train[0]),len(proteins_train[0]))
    plot_model(multi_input_model, to_file='multi_input_model_plot.png', show_shapes=True, show_layer_names=True)
    multi_input_history = multi_input_model.fit([updrsPerPatient_train,proteins_train,peptides_train],updrsPerPatient_train_labels,epochs=900, batch_size=32)
    dlNetwork.plotLoss(multi_input_history)
    multi_input_model.predict([updrsPerPatient_Test,proteins_test,peptides_test])
    updrsPerPatient_Test_labels

#################################################################################################
##  This network Predict updrs for Visit_48 both by priod progression of updrs and peptide abundance
################################################################################################

if(mode==Mode.TWO_INPUT_FC_LSTM):
    updrsPerPatient_train,updrsPerPatient_train_labels, updrsPerPatient_Test,updrsPerPatient_Test_labels = loadInstance.GetUpdrsPerPatient(updrs_visits)
    peptide_train,peptide_tests = loadInstance.GetpeptidePerLastVisit()
    multi_input_model = dlNetwork.build_multi_input_network(len(updrsPerPatient_train[0]))
    plot_model(multi_input_model, to_file='two_input_model_plot.png', show_shapes=True, show_layer_names=True)
    multi_input_history = multi_input_model.fit([updrsPerPatient_train,peptide_train],updrsPerPatient_train_labels,epochs=900, batch_size=32)
    dlNetwork.plotLoss(multi_input_history)
    multi_input_model.predict([updrsPerPatient_Test,peptide_tests])
    updrsPerPatient_Test_labels

###################################################################################
##  This network Predict updrs for Visit_48 by prior parkinson updrs progression V_0,V_3.....V_48
##################################################################################

if(mode==Mode.GRU):
    updrsPerPatient_train,updrsPerPatient_train_labels, updrsPerPatient_Test,updrsPerPatient_Test_labels = loadInstance.GetUpdrsPerPatient(updrs_visits)
    gru_model = dlNetwork.build_GRU_network(len(updrsPerPatient_train[0]))
    gru_history = gru_model.fit(updrsPerPatient_train,updrsPerPatient_train_labels,epochs=205, validation_split=0.2, verbose=1)
    dlNetwork.plotLoss(gru_history)
    gru_model.predict(updrsPerPatient_Test)
    updrsPerPatient_Test_labels

###################################################################################
##  This network Predict updrs for all patian visits by peptide abundance vector data
##################################################################################

if(mode==Mode.FULLY_CONNECTED):
    
    peptide_train, peptide_test, updrs_train, updrs_test =loadInstance.GetTrainAndTestSets(common_visits)
    Fc_model = dlNetwork.buildFullyConnectedNetwork()
    plot_model(Fc_model, to_file='FC_model_plot.png', show_shapes=True, show_layer_names=True)
    fc_history = Fc_model.fit(peptide_train, updrs_train, epochs=85, validation_split=0.2, verbose=1)
    dlNetwork.plotLoss(fc_history)
    Fc_model.predict(peptide_test[:10])
    updrs_test[:10]

