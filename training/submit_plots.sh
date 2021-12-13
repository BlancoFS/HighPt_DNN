#!/bin/bash
cd /gpfs/users/mantecap/CMSSW_11_1_0/src/DNN_optimization
eval `scramv1 runtime -sh`
python doPlots.py
