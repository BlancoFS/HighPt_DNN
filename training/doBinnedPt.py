import pandas as pd
import glob
import argparse


# Read .csv file with samples and add a new column with data divided in pt bins (0, 1, 2, 3) for pt lower than 1150 GeV, between 1150 and 2100 GeV, 
# between 2100 and 3050 GeV and higher than 3050 GeV.

if __name__ == '__main__':
  
  sample = pd.read_csv("train.csv")
  
  sample.loc[sample['Muon_TunePTrack_pt'] > 0, 'binnedPt'] = 4
  sample.loc[sample['Muon_TunePTrack_pt'] < 3050, 'binnedPt'] = 3
  sample.loc[sample['Muon_TunePTrack_pt'] < 2100, 'binnedPt'] = 2
  sample.loc[sample['Muon_TunePTrack_pt'] < 1150, 'binnedPt'] = 1
  
  sample.to_csv('train_binnedPt.csv')
  

  
  

  
  
  

