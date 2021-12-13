#!/bin/sh

cd /gpfs/users/mantecap/CMSSW_11_1_0/src/processing
source /cvmfs/cms.cern.ch/cmsset_default.sh
eval `scramv1 runtime -sh`

python doStep2.py --FileNumber $1 --kind minus_barrel_pu #554
#python doStep2.py --FileNumber $1 --kind minus_barrel_nopu #97
# python doStep2.py --FileNumber $1 --kind minus_pos_pu #293
# python doStep2.py --FileNumber $1 --kind minus_pos_nopu #49
# python doStep2.py --FileNumber $1 --kind minus_neg_pu #294
# python doStep2.py --FileNumber $1 --kind minus_neg_nopu #51

#python doStep2.py --FileNumber $1 --kind plus_barrel_pu #530
#python doStep2.py --FileNumber $1 --kind plus_barrel_nopu #96
# python doStep2.py --FileNumber $1 --kind plus_pos_pu #284
# python doStep2.py --FileNumber $1 --kind plus_pos_nopu #51
# python doStep2.py --FileNumber $1 --kind plus_neg_pu #290
# python doStep2.py --FileNumber $1 --kind plus_neg_nopu #52
