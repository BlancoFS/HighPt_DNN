#!/bin/sh

cd /gpfs/users/blancose/HighPT/CMSSW_11_0_3/src/HighPt_DNN/training
source /cvmfs/cms.cern.ch/cmsset_default.sh
eval `scramv1 runtime -sh`

python doBinnedPt.py
