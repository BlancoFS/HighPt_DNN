#!/bin/bash                                                                                                                                                                                                \
                                                                                                                                                                                                            
cd /gpfs/users/blancose/HighPT/CMSSW_11_0_3/src
eval `scramv1 runtime -sh`
cd /gpfs/users/blancose/HighPT/CMSSW_11_0_3/src/HighPt_DNN/training

python doFits.py
