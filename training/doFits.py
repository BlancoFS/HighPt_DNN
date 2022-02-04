from keras import models
from scipy import stats

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
#import matplotlib.pyplot as plt                                                                                                                                                                           
import pandas as pd
import numpy as np
import ROOT


#test = pd.read_csv('train_barrel_pu.csv')                                                                                                                                                                 
#test = pd.read_csv('test.csv')                                                                                                                                                   
test = pd.read_csv('test_binnedPt.csv')

s = str(sys.argv[1])

model = models.load_model(s)


#### Non parametrized neural network // BEST MODELS                                                                                                                                                        

#model = models.load_model('Models_DNN/model_nFirstNeurons256_nHiddenLayers_5_LearningRate_0.000428422182649_BatchSize_256.h5')                                                                     
#model = models.load_model('model_nFirstNeurons512_nHiddenLayers_10_LearningRate_0.000472547551122_BatchSize_2048.h5')                                                                                     

#### Parametrized neural network // BEST MODELS                                                                                                                                                            

#model = models.load_model('model_nFirstNeurons128_nHiddenLayers_6_LearningRate_0.000173913904526_BatchSize_256.h5')

### Non parametrized variables

#variablesTrain = ["Muon_InnerTrack_eta", "Muon_InnerTrack_phi", "Muon_InnerTrack_charge", "Muon_InnerTrack_pt",  "Muon_TunePTrack_pt", "Muon_DT_s1_nhits","Muon_DT_s1_x_mean","Muon_DT_s1_y_mean","Muon_DT_s1_z_mean","Muon_DT_s1_x_std","Muon_DT_s1_y_std","Muon_DT_s1_z_std","Muon_DT_s1_x_skew","Muon_DT_s1_y_skew","Muon_DT_s1_z_skew","Muon_DT_s1_x_kurt","Muon_DT_s1_y_kurt","Muon_DT_s1_z_kurt","Muon_DT_s2_nhits","Muon_DT_s2_x_mean","Muon_DT_s2_y_mean","Muon_DT_s2_z_mean","Muon_DT_s2_x_std","Muon_DT_s2_y_std","Muon_DT_s2_z_std","Muon_DT_s2_x_skew","Muon_DT_s2_y_skew","Muon_DT_s2_z_skew","Muon_DT_s2_x_kurt","Muon_DT_s2_y_kurt","Muon_DT_s2_z_kurt","Muon_DT_s3_nhits","Muon_DT_s3_x_mean","Muon_DT_s3_y_mean","Muon_DT_s3_z_mean","Muon_DT_s3_x_std","Muon_DT_s3_y_std","Muon_DT_s3_z_std","Muon_DT_s3_x_skew","Muon_DT_s3_y_skew","Muon_DT_s3_z_skew","Muon_DT_s3_x_kurt","Muon_DT_s3_y_kurt","Muon_DT_s3_z_kurt","Muon_DT_s4_nhits","Muon_DT_s4_x_mean","Muon_DT_s4_y_mean","Muon_DT_s4_x_std","Muon_DT_s4_y_std","Muon_DT_s4_x_skew","Muon_DT_s4_y_skew","Muon_DT_s4_x_kurt","Muon_DT_s4_y_kurt"]                                                                                                                                        

### Parametrized variables

# Muon_InnerTrack_charge not included

variablesTrain = ["Muon_InnerTrack_eta", "Muon_InnerTrack_phi", "Muon_InnerTrack_charge", "Muon_InnerTrack_pt",  "Muon_TunePTrack_pt", "Muon_DT_s1_nhits","Muon_DT_s1_x_mean","Muon_DT_s1_y_mean","Muon_DT_s1_z_mean","Muon_DT_s1_x_std","Muon_DT_s1_y_std","Muon_DT_s1_z_std","Muon_DT_s1_x_skew","Muon_DT_s1_y_skew","Muon_DT_s1_z_skew","Muon_DT_s1_x_kurt","Muon_DT_s1_y_kurt","Muon_DT_s1_z_kurt","Muon_DT_s2_nhits","Muon_DT_s2_x_mean","Muon_DT_s2_y_mean","Muon_DT_s2_z_mean","Muon_DT_s2_x_std","Muon_DT_s2_y_std","Muon_DT_s2_z_std","Muon_DT_s2_x_skew","Muon_DT_s2_y_skew","Muon_DT_s2_z_skew","Muon_DT_s2_x_kurt","Muon_DT_s2_y_kurt","Muon_DT_s2_z_kurt","Muon_DT_s3_nhits","Muon_DT_s3_x_mean","Muon_DT_s3_y_mean","Muon_DT_s3_z_mean","Muon_DT_s3_x_std","Muon_DT_s3_y_std","Muon_DT_s3_z_std","Muon_DT_s3_x_skew","Muon_DT_s3_y_skew","Muon_DT_s3_z_skew","Muon_DT_s3_x_kurt","Muon_DT_s3_y_kurt","Muon_DT_s3_z_kurt","Muon_DT_s4_nhits","Muon_DT_s4_x_mean","Muon_DT_s4_y_mean","Muon_DT_s4_x_std","Muon_DT_s4_y_std","Muon_DT_s4_x_skew","Muon_DT_s4_y_skew","Muon_DT_s4_x_kurt","Muon_DT_s4_y_kurt", "binnedPt"]


test = test[(test.Muon_TunePTrack_pt < 10000)]

Rreco = (test.Muon_Genpt.values-test.Muon_TunePTrack_pt.values)/test.Muon_Genpt.values
Rpred = (test.Muon_Genpt.values-model.predict(test[variablesTrain]).ravel())/test.Muon_Genpt
pTpred = model.predict(test[variablesTrain]).ravel()
tunePpT = test.Muon_TunePTrack_pt.values
genpT = test.Muon_Genpt.values
RelRreco = np.std(abs(test.Muon_Genpt.values-test.Muon_TunePTrack_pt.values)/test.Muon_Genpt.values)
RelPred = np.std(abs(test.Muon_Genpt.values-model.predict(test[variablesTrain]).ravel())/test.Muon_Genpt)


#### 2D PLOTS TuneP vs DNN

plt.figure(figsize=(15,10)) 
plt.hist2d(genpT, tunePpT, bins=[50,50] ,range=[[200, 4000], [200, 8000]], norm=mpl.colors.LogNorm())
plt.xlabel('GenpT [GeV]',fontsize=14)
plt.ylabel('TuneP_pT [GeV]',fontsize=14)
plt.tick_params(axis='both', labelsize=13)

#legend
clb = plt.colorbar()
clb.set_label('nMuons', fontsize=15)
clb.ax.tick_params(labelsize=13)

plt.savefig('data_test_tuneppt_genpt.png')
plt.clf()
plt.cla()


plt.figure(figsize=(15,10)) 
plt.hist2d(genpT, pTpred, bins=[50,50] ,range=[[200, 4000], [200, 8000]], norm=mpl.colors.LogNorm())
plt.xlabel('GenpT [GeV]',fontsize=14)
plt.ylabel('Predicted pT [GeV]',fontsize=14)
plt.tick_params(axis='both', labelsize=13)


#legend
clb = plt.colorbar()
clb.set_label('nMuons', fontsize=15)
clb.ax.tick_params(labelsize=13)


plt.savefig('data_test_ptpred_genpt.png')
plt.clf()
plt.cla()



#### First Histogram: Full pT range

R_reco_full = ROOT.TH1F("R_reco_full", "R_reco_full", 200, -0.2, 0.2)
R_pred_full = ROOT.TH1F("R_pred_full", "R_pred_full", 200, -0.2, 0.2)

for i in Rreco:
    R_reco_full.Fill(i)

for j in Rpred:
    R_pred_full.Fill(j)


canvas_full = ROOT.TCanvas("canvas1", "canvas1", 900, 800)
canvas_full.SetBottomMargin(0.24)

gauss_reco = ROOT.TF1("gauss_reco", "gaus")
gauss_pred = ROOT.TF1("gauss_pred", "gaus")


R_reco_full.Fit(gauss_reco)
R_pred_full.Fit(gauss_pred)

R_reco_full.SetLineColor(ROOT.kRed+1)
R_reco_full.GetFunction("gauss_reco").SetLineColor(ROOT.kRed)

R_pred_full.SetLineColor(ROOT.kBlue+1)
R_pred_full.GetFunction("gauss_pred").SetLineColor(ROOT.kBlue)


R_reco_full.SetStats(False)
R_pred_full.SetStats(False)
ROOT.gStyle.SetOptFit(False)

R_reco_full.SetTitle("")
R_pred_full.SetTitle("")

R_reco_full.GetYaxis().SetTitle('Events')
R_reco_full.GetYaxis().SetTitleOffset(1.5)
R_reco_full.GetXaxis().SetTitle('R = #frac{p_{T}^{gen} - p_{T}}{p_{T}^{gen}}')
R_reco_full.GetXaxis().SetTitleOffset(2.33)


ROOT.gStyle.SetPaintTextFormat("5.3f")

max1 = R_reco_full.GetMaximum()
max2 = R_pred_full.GetMaximum()

if max1 > max2:
    R_reco_full.SetMaximum(max1*0.2 + max1)
    R_pred_full.SetMaximum(max1*0.2 + max1)
else:
    R_reco_full.SetMaximum(max2*0.2 + max2)
    R_pred_full.SetMaximum(max2*0.2 + max2)

R_reco_full.Draw()
R_pred_full.Draw("same")

legend_full = ROOT.TLegend(0.9, 0.89, 0.7, 0.78)

legend_full.AddEntry(R_reco_full, "TuneP", "l")
legend_full.AddEntry(gauss_reco, "Mean = " + str(round(gauss_reco.GetParameter(1), 4)) + "; #sigma = " + str(round(gauss_reco.GetParameter(2), 2)), "l")
legend_full.AddEntry(R_pred_full, "DNN Model", "l")
legend_full.AddEntry(gauss_pred, "Mean = " + str(round(gauss_pred.GetParameter(1), 4)) + "; #sigma = " + str(round(gauss_pred.GetParameter(2), 2)), "l")

legend_full.SetTextFont(42)
legend_full.SetBorderSize(0)
legend_full.SetFillColor(0)
legend_full.Draw()

canvas_full.Update()

canvas_full.Draw()
canvas_full.SaveAs("Fits/c_TuneP_vs_Model_R.pdf")



### MAKE PLOTS IN PT RANGE: 200, 500, 800, 1100, 1400, 1700, 2000, 2300, 2600, 2900, 4000 

pt_bins = [200, 500, 800, 1100, 1400, 1700, 2000, 2300, 2600, 2900, 3200, 3500, 3800, 4000]
#pt_bins = [200, 500, 800, 1100, 1400, 1700, 2000, 2300, 2600, 2900, 4000]

mean_values = []
std_values = []
mean_reco_values = []
std_reco_values = []
x = []
ex = []
mean_err = []
std_err = []
mean_reco_err = []
std_reco_err = []

for i in range (0, len(pt_bins)-1):

    low_pt = pt_bins[i]
    high_pt = pt_bins[i+1]
    
    x.append((low_pt+high_pt)/2)
    ex.append((high_pt-low_pt)/2)

    tmp = test.loc[test.Muon_Genpt>low_pt]
    tmp = tmp.loc[tmp.Muon_Genpt<high_pt]

    Rreco = (tmp.Muon_Genpt.values-tmp.Muon_TunePTrack_pt.values)/tmp.Muon_Genpt.values
    Rpred = (tmp.Muon_Genpt.values-model.predict(tmp[variablesTrain]).ravel())/tmp.Muon_Genpt

    canvas = ROOT.TCanvas("canvas" + str(high_pt), "canvas" + str(high_pt), 900, 800)
    canvas.SetBottomMargin(0.24)
    
    R_reco = ROOT.TH1F("R_reco_" + str(high_pt), "R_reco_" + str(high_pt), 200, -0.2, 0.2)
    R_pred = ROOT.TH1F("R_pred_" + str(high_pt), "R_pred_" + str(high_pt), 200, -0.2, 0.2)
    
    for i in Rreco:
        R_reco.Fill(i)
        
    for j in Rpred:
        R_pred.Fill(j)

    gauss_reco = ROOT.TF1("gauss_reco", "gaus")
    gauss_pred = ROOT.TF1("gauss_pred", "gaus")
    
    R_reco.Fit(gauss_reco)
    R_pred.Fit(gauss_pred)
    
    R_reco.SetLineColor(ROOT.kRed+1)
    R_reco.GetFunction("gauss_reco").SetLineColor(ROOT.kRed)
    
    R_pred.SetLineColor(ROOT.kBlue+1)
    R_pred.GetFunction("gauss_pred").SetLineColor(ROOT.kBlue)
    gauss_pred.SetLineColor(ROOT.kRed)
    
    R_reco.SetStats(False)
    R_pred.SetStats(False)
    ROOT.gStyle.SetOptFit(False)
    
    R_reco.SetTitle(str(low_pt) + "-" + str(high_pt) +"GeV")
    R_pred.SetTitle("")

    R_reco.GetYaxis().SetTitle('Events')
    R_reco.GetYaxis().SetTitleOffset(1.5)
    R_reco.GetXaxis().SetTitle('R = #frac{p_{T}^{gen} - p_{T}}{p_{T}^{gen}}')
    R_reco.GetXaxis().SetTitleOffset(2.33)
    
    ROOT.gStyle.SetPaintTextFormat("5.3f")

    max1 = R_reco.GetMaximum()
    max2 = R_pred.GetMaximum()
    
    if max1 > max2:
        R_reco.SetMaximum(max1*0.2 + max1)
        R_pred.SetMaximum(max1*0.2 + max1)
    else:
        R_reco.SetMaximum(max2*0.2 + max2)
        R_pred.SetMaximum(max2*0.2 + max2)
    
    R_reco.Draw()
    R_pred.Draw("same")
    
    legend = ROOT.TLegend(0.9, 0.89, 0.7, 0.78)
    legend.AddEntry(R_reco, "TuneP", "l")
    legend.AddEntry(gauss_reco, "Mean = " + str(round(gauss_reco.GetParameter(1), 4)) + "; #sigma = " + str(round(gauss_reco.GetParameter(2), 2)), "l")
    legend.AddEntry(R_pred, "DNN Model", "l")
    legend.AddEntry(gauss_pred, "Mean = " + str(round(gauss_pred.GetParameter(1), 4)) + "; #sigma = " + str(round(gauss_pred.GetParameter(2), 2)), "l")

    legend.SetTextFont(42)
    legend.SetBorderSize(0)
    legend.SetFillColor(0)
    legend.Draw()

    canvas.Draw()
    canvas.SaveAs("/gpfs/users/blancose/HighPT/CMSSW_11_0_3/src/HighPt_DNN/training/Fits/c_TuneP_vs_Model_R_" + str(low_pt) + "_" + str(high_pt) + ".pdf")
    
    mean_reco_values.append(gauss_reco.GetParameter(1))
    std_reco_values.append(gauss_reco.GetParameter(2))
    mean_values.append(gauss_pred.GetParameter(1))
    std_values.append(gauss_pred.GetParameter(2))
    mean_err.appen(gauss_pred.GetParError(1))
    std_err.append(gauss_pred.GetParError(2))
    mean_reco_err.appen(gauss_reco.GetParError(1))
    std_reco_err.append(gauss_reco.GetParError(2))
    
    
    
plt.errorbar(x, Relreco, yerr=std_reco_err, xerr=ex , linewidth=5, fmt='o')
plt.errorbar(x, Relpred, yerr=std_err, xerr=ex , linewidth=5, fmt='o')

plt.xlabel('Muon GenpT [GeV]',fontsize=20)
plt.ylabel('$\sigma_{R}$',fontsize=26)
plt.tick_params(axis='both', labelsize=20)
plt.legend(['TuneP', 'DNN'], loc='upper left')
plt.savefig('SigmaR_vs_genpT.png')

plt.clf()
plt.cla()

plt.errorbar(x, mean_reco_values, yerr=mean_reco_err, xerr=ex , linewidth=5, fmt='o')
plt.errorbar(x, mean_values, yerr=mean_err, xerr=ex , linewidth=5, fmt='o')


plt.xlabel('Muon GenpT [GeV]',fontsize=20)
plt.ylabel('$\mu_{R}$',fontsize=26)
plt.tick_params(axis='both', labelsize=20)
plt.legend(['TuneP', 'DNN'], loc='upper left')
plt.savefig('MeanR_vs_genpT.png')

plt.clf()
plt.cla()
    
