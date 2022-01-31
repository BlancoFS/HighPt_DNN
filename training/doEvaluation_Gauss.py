import os, sys, stat 
from keras import models
import pandas as pd
import ROOT

test = pd.read_csv('test_binnedPt.csv')

s = str(sys.argv[1])

model = models.load_model(s)

variablesTrain = ["Muon_InnerTrack_eta", "Muon_InnerTrack_phi", "Muon_InnerTrack_charge", "Muon_InnerTrack_pt",  "Muon_TunePTrack_pt", "Muon_DT_s1_nhits","Muon_DT_s1_x_mean","Muon_DT_s1_y_mean","Muon_DT_s1_z_mean","Muon_DT_s1_x_std","Muon_DT_s1_y_std","Muon_DT_s1_z_std","Muon_DT_s1_x_skew","Muon_DT_s1_y_skew","Muon_DT_s1_z_skew","Muon_DT_s1_x_kurt","Muon_DT_s1_y_kurt","Muon_DT_s1_z_kurt","Muon_DT_s2_nhits","Muon_DT_s2_x_mean","Muon_DT_s2_y_mean","Muon_DT_s2_z_mean","Muon_DT_s2_x_std","Muon_DT_s2_y_std","Muon_DT_s2_z_std","Muon_DT_s2_x_skew","Muon_DT_s2_y_skew","Muon_DT_s2_z_skew","Muon_DT_s2_x_kurt","Muon_DT_s2_y_kurt","Muon_DT_s2_z_kurt","Muon_DT_s3_nhits","Muon_DT_s3_x_mean","Muon_DT_s3_y_mean","Muon_DT_s3_z_mean","Muon_DT_s3_x_std","Muon_DT_s3_y_std","Muon_DT_s3_z_std","Muon_DT_s3_x_skew","Muon_DT_s3_y_skew","Muon_DT_s3_z_skew","Muon_DT_s3_x_kurt","Muon_DT_s3_y_kurt","Muon_DT_s3_z_kurt","Muon_DT_s4_nhits","Muon_DT_s4_x_mean","Muon_DT_s4_y_mean","Muon_DT_s4_x_std","Muon_DT_s4_y_std","Muon_DT_s4_x_skew","Muon_DT_s4_y_skew","Muon_DT_s4_x_kurt","Muon_DT_s4_y_kurt", "binnedPt"]

model_loss = model.evaluate(test[variablesTrain], test['Muon_Genpt'])

start = s.find('Neurons') + 7
end = s.find('_nHiddenLayers', start)
nneurons = s[start:end]

start = s.find('Layers_') + 7
end = s.find('_LearningRate', start)
nlayers = s[start:end]

start = s.find('ngRate_') + 7
end = s.find('_BatchSize', start)
lr = s[start:end]

start = s.find('chSize_') + 7
end = s.find('.h5', start)
batch = s[start:end]


### Fit a gaussian distribution to the DNN predicted pT in the last bin 3500-4000 GeV


tmp = test.loc[test.Muon_Genpt>3500]

Rpred = (tmp.Muon_Genpt.values-model.predict(tmp[variablesTrain]).ravel())/tmp.Muon_Genpt

R_pred = ROOT.TH1F("R_pred_" + str(high_pt), "R_pred_" + str(high_pt), 300, -0.4, 0.4)

for j in Rpred:
        R_pred.Fill(j)

gauss_pred = ROOT.TF1("gauss_pred", "gaus")
R_pred.Fit(gauss_pred)

### Save the mean and std to check the existence of a bias

f = open("evaluation.txt", "a")
f.write(nneurons + "\t" + nlayers + "\t" + lr + "\t" + batch + "\t" + str(model_loss) + "\t" + str(gauss_pred.GetParameter(1)) + "\t" + str(gauss_pred.GetParameter(2)) +  "\n")
f.close
