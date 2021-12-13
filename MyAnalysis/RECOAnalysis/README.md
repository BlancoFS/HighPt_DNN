# Muon high-pT variables ntupler (RECO format)

This framework is devoted to get the relevant collections from RECO-SIM and put them into plain trees.

## Installation

cmsrel CMSSW_11_0_3

cd CMSSW_11_0_3/src

cmsenv

git clone git@github.com:fmanteca/HighPt_DNN.git

cd MyAnalysis

scram b -j 10

cd RECOAnalysis/test

cmsRun runRECOAnalysis_cfg.py


