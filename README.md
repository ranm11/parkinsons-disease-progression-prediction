# parkinsons-disease-progression-prediction
predict the Parkinsons disease progression by proteins and peptides  
the goad of this competition is to predict UDPRS for monthes 6 , 12 ,24
get data from :
https://www.kaggle.com/competitions/amp-parkinsons-disease-progression-prediction/data



sample data :

PS G:\Old_Disk\Documents\Develpos\keras\parkinsons-disease-progression-prediction\amp-parkinsons-disease-progression-prediction> cat .\train_clinical_data.csv |select -First 10
visit_id,patient_id,visit_month,updrs_1,updrs_2,updrs_3,updrs_4,upd23b_clinical_state_on_medication
55_0,55,0,10,6,15,,
55_3,55,3,10,7,25,,
55_6,55,6,8,10,34,,
55_9,55,9,8,9,30,0,On
55_12,55,12,10,10,41,0,On
55_18,55,18,7,13,38,0,On
55_24,55,24,16,9,49,0,On
55_30,55,30,14,13,49,0,On
55_36,55,36,17,18,51,0,On

PS G:\Old_Disk\Documents\Develpos\keras\parkinsons-disease-progression-prediction\amp-parkinsons-disease-progression-prediction> cat .\train_peptides.csv |select -First 10
visit_id,visit_month,patient_id,UniProt,Peptide,PeptideAbundance
55_0,0,55,O00391,NEQEQPLGQWHLS,11254.3
55_0,0,55,O00533,GNPEPTFSWTK,102060.0
55_0,0,55,O00533,IEIPSSVQQVPTIIK,174185.0
55_0,0,55,O00533,KPQSAVYSTGSNGILLC(UniMod_4)EAEGEPQPTIK,27278.9
55_0,0,55,O00533,SMEQNGPGLEYR,30838.7
55_0,0,55,O00533,TLKIENVSYQDKGNYR,23216.5
55_0,0,55,O00533,VIAVNEVGR,170878.0
55_0,0,55,O00533,VMTPAVYAPYDVK,148771.0
55_0,0,55,O00533,VNGSPVDNHPFAGDVVFPR,55202.1

PS G:\Old_Disk\Documents\Develpos\keras\parkinsons-disease-progression-prediction\amp-parkinsons-disease-progression-prediction> cat .\train_proteins.csv |select -First 10
visit_id,visit_month,patient_id,UniProt,NPX
55_0,0,55,O00391,11254.3
55_0,0,55,O00533,732430.0
55_0,0,55,O00584,39585.8
55_0,0,55,O14498,41526.9
55_0,0,55,O14773,31238.0
55_0,0,55,O14791,4202.71
55_0,0,55,O15240,177775.0
55_0,0,55,O15394,62898.2
55_0,0,55,O43505,333376.0
