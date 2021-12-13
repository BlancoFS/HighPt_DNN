import ROOT
import argparse
from math import *
import os

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

    out_hits = open('data_' + args.kind + '/output_Hit_' + args.fileNumber + '.txt', 'w')
    out_muons = open('data_' + args.kind + '/output_Muon_' + args.fileNumber + '.txt', 'w')
    out_props = open('data_' + args.kind + '/output_Prop_' + args.fileNumber + '.txt', 'w')
    
    rootfile = ROOT.TFile.Open(args.inputDir + 'tree_' + args.fileNumber + '.root', "READ")
    tree = rootfile.Get("Events")
            
    for event in tree:

        for entry in range(0,eval('event.' + variables_muons[0] + '.size()')):
                
                #print str(entry)
            line = str(eval('event.' + variables_muons[0] + '.at(' + str(entry) + ')')) + '\t' + str(eval('event.' + variables_muons[1] + '.at(' + str(entry) + ')')) + '\t' + str(eval('event.' + variables_muons[2] + '.at(' + str(entry) + ')')) + '\t' + str(eval('event.' + variables_muons[3] + '.at(' + str(entry) + ')')) + '\t' + str(eval('event.' + variables_muons[4] + '.at(' + str(entry) + ')')) + '\t' + str(eval('event.' + variables_muons[5] + '.at(' + str(entry) + ')')) + '\t' + str(eval('event.' + variables_muons[6] + '.at(' + str(entry) + ')')) + '\t' + str(eval('event.' + variables_muons[7] + '.at(' + str(entry) + ')')) + '\t' + str(eval('event.' + variables_muons[8] + '.at(' + str(entry) + ')')) + '\t' + str(eval('event.' + variables_muons[9] + '.at(' + str(entry) + ')')) + '\t' + str(eval('event.' + variables_muons[10] + '.at(' + str(entry) + ')')) + '\t' + str(eval('event.' + variables_muons[11] + '.at(' + str(entry) + ')')) + '\t' + str(eval('event.' + variables_muons[12] + '.at(' + str(entry) + ')')) + '\t' + str(eval('event.' + variables_muons[13] + '.at(' + str(entry) + ')')) + '\t' + str(eval('event.' + variables_muons[14] + '.at(' + str(entry) + ')')) + '\t' + str(eval('event.' + variables_muons[15] + '.at(' + str(entry) + ')')) + '\t' + str(eval('event.' + variables_muons[16] + '.at(' + str(entry) + ')')) + '\t' + str(eval('event.' + variables_muons[17] + '.at(' + str(entry) + ')')) + '\t' + str(eval('event.' + variables_muons[18] + '.at(' + str(entry) + ')')) + '\t' + str(eval('event.' + variables_muons[19] + '.at(' + str(entry) + ')')) + '\t' + str(eval('event.' + variables_muons[20] + '.at(' + str(entry) + ')')) + '\t' + str(eval('event.' + variables_muons[21] + '.at(' + str(entry) + ')')) + '\t' + str(eval('event.' + variables_muons[22] + '.at(' + str(entry) + ')')) + '\t' + str(eval('event.' + variables_muons[23] + '.at(' + str(entry) + ')')) + '\t' + str(eval('event.' + variables_muons[24] + '.at(' + str(entry) + ')')) + '\t' + str(eval('event.' + variables_muons[25] + '.at(' + str(entry) + ')')) + '\t' + str(eval('event.' + variables_muons[26] + '.at(' + str(entry) + ')')) + '\t' + str(eval('event.' + variables_muons[27] + '.at(' + str(entry) + ')')) + '\t' + str(eval('event.' + variables_muons[28] + '.at(' + str(entry) + ')')) + '\t' + str(eval('event.' + variables_muons[29] + '.at(' + str(entry) + ')')) + '\t' + str(eval('event.' + variables_muons[30] + '.at(' + str(entry) + ')'))
                
            out_muons.write(line + '\n')


        for entry in range(0,eval('event.' + variables_props[0] + '.size()')):
                
                #print str(entry)
            line = str(eval('event.' + variables_props[0] + '.at(' + str(entry) + ')')) + '\t' + str(eval('event.' + variables_props[1] + '.at(' + str(entry) + ')')) + '\t' + str(eval('event.' + variables_props[2] + '.at(' + str(entry) + ')')) + '\t' + str(eval('event.' + variables_props[3] + '.at(' + str(entry) + ')')) + '\t' + str(eval('event.' + variables_props[4] + '.at(' + str(entry) + ')')) + '\t' + str(eval('event.' + variables_props[5] + '.at(' + str(entry) + ')')) + '\t' + str(eval('event.' + variables_props[6] + '.at(' + str(entry) + ')')) + '\t' + str(eval('event.' + variables_props[7] + '.at(' + str(entry) + ')')) + '\t' + str(eval('event.' + variables_props[8] + '.at(' + str(entry) + ')')) + '\t' + str(eval('event.' + variables_props[9] + '.at(' + str(entry) + ')')) + '\t' + str(eval('event.' + variables_props[10] + '.at(' + str(entry) + ')')) + '\t' + str(eval('event.' + variables_props[11] + '.at(' + str(entry) + ')'))
                
            out_props.write(line + '\n')

                
        for entry in range(0,eval('event.' + variables_hits[0] + '.size()')):
                
                #print str(entry)
            line = str(eval('event.' + variables_hits[0] + '.at(' + str(entry) + ')')) + '\t' + str(eval('event.' + variables_hits[1] + '.at(' + str(entry) + ')')) + '\t' + str(eval('event.' + variables_hits[2] + '.at(' + str(entry) + ')')) + '\t' + str(eval('event.' + variables_hits[3] + '.at(' + str(entry) + ')')) + '\t' + str(eval('event.' + variables_hits[4] + '.at(' + str(entry) + ')')) + '\t' + str(eval('event.' + variables_hits[5] + '.at(' + str(entry) + ')')) + '\t' + str(eval('event.' + variables_hits[6] + '.at(' + str(entry) + ')')) + '\t' + str(eval('event.' + variables_hits[7] + '.at(' + str(entry) + ')')) + '\t' + str(eval('event.' + variables_hits[8] + '.at(' + str(entry) + ')')) + '\t' + str(eval('event.' + variables_hits[9] + '.at(' + str(entry) + ')')) + '\t' + str(eval('event.' + variables_hits[10] + '.at(' + str(entry) + ')')) + '\t' + str(eval('event.' + variables_hits[11] + '.at(' + str(entry) + ')')) + '\t' + str(eval('event.' + variables_hits[12] + '.at(' + str(entry) + ')')) + '\t' + str(eval('event.' + variables_hits[13] + '.at(' + str(entry) + ')')) + '\t' + str(eval('event.' + variables_hits[14] + '.at(' + str(entry) + ')')) + '\t' + str(eval('event.' + variables_hits[15] + '.at(' + str(entry) + ')')) + '\t' + str(eval('event.' + variables_hits[16] + '.at(' + str(entry) + ')')) + '\t' + str(eval('event.' + variables_hits[17] + '.at(' + str(entry) + ')'))  + '\t' + str(eval('event.' + variables_hits[18] + '.at(' + str(entry) + ')')) + '\t' + str(eval('event.' + variables_hits[19] + '.at(' + str(entry) + ')'))
                
            out_hits.write(line + '\n')
        


    out_hits.close()
    out_muons.close()
    out_props.close()
    rootfile.Close()
