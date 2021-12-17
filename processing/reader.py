import ROOT
import argparse
from math import *
import os
import uproot
import pandas as pd
import numpy as np

# python reader.py --inputDir /gpfs/projects/cms/fernanpe/ZprimeToMuMu_M-5000_TuneCP5_13TeV-madgraphMLM-pythia8/ZprimeToMuMu_M-5000_ntupler/200630_000605/0000/ --part 1

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description = "Receive the parameters")
    parser.add_argument('--inputDir', action = 'store', type = str, dest = 'inputDir', help = 'Define the inputDir path')
    parser.add_argument('--fileNumber', action = 'store', type = str, dest = 'fileNumber', help = 'files number')
    parser.add_argument('--kind', action = 'store', type = str, dest = 'kind', help = 'Kind of sample')

    args = parser.parse_args()

    variables_muons = ['Muon_Eventid', 'Muon_EventluminosityBlock', 'Muon_Muonid', 'Muon_nHits', 'Muon_nGeomDets', 'Muon_Genpt', 'Muon_InnerTrack_pt', 'Muon_InnerTrack_eta', 'Muon_InnerTrack_phi', 'Muon_InnerTrack_charge', 'Muon_InnerTrack_ptErr', 'Muon_InnerTrack_Chindf', 'Muon_TunePTrack_pt', 'Muon_TunePTrack_ptErr', 'Muon_nShowers', 'Muon_hasShowerInStation_DT_1', 'Muon_hasShowerInStation_DT_2', 'Muon_hasShowerInStation_DT_3' , 'Muon_hasShowerInStation_DT_4', 'Muon_hasShowerInStation_CSC_1', 'Muon_hasShowerInStation_CSC_2', 'Muon_hasShowerInStation_CSC_3' , 'Muon_hasShowerInStation_CSC_4', 'Muon_nDigisInStation_DT_1', 'Muon_nDigisInStation_DT_2', 'Muon_nDigisInStation_DT_3' , 'Muon_nDigisInStation_DT_4', 'Muon_nDigisInStation_CSC_1', 'Muon_nDigisInStation_CSC_2', 'Muon_nDigisInStation_CSC_3' , 'Muon_nDigisInStation_CSC_4']

    variables_hits = ['Hit_Eventid', 'Hit_EventluminosityBlock', 'Hit_Muonid', 'Hit_Hitid', 'Hit_Detid', 'Hit_isDT', 'Hit_isCSC', 'Hit_DTstation', 'Hit_CSCstation', 'Hit_DetElement', 'Hit_x', 'Hit_y', 'Hit_z', 'Hit_distToProp', 'Hit_Compatibility', 'Hit_dirx', 'Hit_diry', 'Hit_dirz', 'Hit_chi2', 'Hit_ndof']

    variables_props = ['Prop_Eventid', 'Prop_EventluminosityBlock', 'Prop_Muonid', 'Prop_Detid', 'Prop_isDT', 'Prop_isCSC', 'Prop_DTstation', 'Prop_CSCstation', 'Prop_DetElement', 'Prop_x', 'Prop_y', 'Prop_z']

    counter = 0

    
    data_muons = uproot.open(args.inputDir + 'tree_' + args.fileNumber + '.root')['Events'].arrays(variables_muons)
    data_hits = uproot.open(args.inputDir + 'tree_' + args.fileNumber + '.root')['Events'].arrays(variables_hits)
    data_props = uproot.open(args.inputDir + 'tree_' + args.fileNumber + '.root')['Events'].arrays(variables_props)

    muons = pd.DataFrame.from_dict(data_muons)
    hits = pd.DataFrame.from_dict(data_hits)
    props = pd.DataFrame.from_dict(data_props)


    muons.to_csv('data_' + args.kind + '/output_Muon_' + args.fileNumber + '.csv', header=False)
    hits.to_csv('data_' + args.kind + '/output_Hit_' + args.fileNumber + '.csv', header=False)
    props.to_csv('data_' + args.kind + '/output_Prop_' + args.fileNumber + '.csv', header=False)


    
    

    
    
    
    
