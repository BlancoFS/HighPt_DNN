import pandas as pd
import glob
import argparse

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description = "Receive the parameters")
    parser.add_argument('--FileNumber', action = 'store', type = str, dest = 'FileNumber', help = 'Which file number to take as input')
    parser.add_argument('--kind', action = 'store', type = str, dest = 'kind', help = 'Kind of file')

    args = parser.parse_args()


    frame_muons = pd.read_csv('data_' + args.kind + '/output_Muon_' + args.FileNumber + '.txt', sep='\t', header=None)

    frame_muons.columns = ['Muon_Eventid', 'Muon_EventluminosityBlock', 'Muon_Muonid', 'Muon_nHits', 'Muon_nGeomDets', 'Muon_Genpt', 'Muon_InnerTrack_pt', 'Muon_InnerTrack_eta', 'Muon_InnerTrack_phi', 'Muon_InnerTrack_charge', 'Muon_InnerTrack_ptErr', 'Muon_InnerTrack_Chindf', 'Muon_TunePTrack_pt', 'Muon_TunePTrack_ptErr','Muon_nShowers', 'Muon_hasShowerInStation_DT_1', 'Muon_hasShowerInStation_DT_2', 'Muon_hasShowerInStation_DT_3' , 'Muon_hasShowerInStation_DT_4', 'Muon_hasShowerInStation_CSC_1', 'Muon_hasShowerInStation_CSC_2', 'Muon_hasShowerInStation_CSC_3' , 'Muon_hasShowerInStation_CSC_4', 'Muon_nDigisInStation_DT_1', 'Muon_nDigisInStation_DT_2', 'Muon_nDigisInStation_DT_3' , 'Muon_nDigisInStation_DT_4', 'Muon_nDigisInStation_CSC_1', 'Muon_nDigisInStation_CSC_2', 'Muon_nDigisInStation_CSC_3' , 'Muon_nDigisInStation_CSC_4']

    all_files = glob.glob('STEP1_' + args.kind + '/*.csv')

    frame_hits = pd.read_csv('STEP1_' + args.kind + '/output' + args.FileNumber + '.csv')


    merge1 = pd.merge(frame_muons, frame_hits, on=['Muon_Eventid','Muon_EventluminosityBlock','Muon_Muonid'])
    del frame_muons, frame_hits


    frame_muonMeans = pd.read_csv('CLEANED_SEGMENTS_' + args.kind + '/cleaned_mean_' + args.FileNumber + '.csv')
    
    output = pd.merge(merge1, frame_muonMeans, on=['Muon_Eventid','Muon_EventluminosityBlock','Muon_Muonid'])
    del merge1, frame_muonMeans
    output.to_csv('train_' + args.kind + '/TrainFile' + args.FileNumber + '.csv', index=False, header=False) #OJO EL HEADER! (para ver nombres de columnas)

