#!/bin/sh

cd /gpfs/users/mantecap/CMSSW_11_1_0/src/processing/
source /cvmfs/cms.cern.ch/cmsset_default.sh
eval `scramv1 runtime -sh`

#python reader.py --inputDir /gpfs/projects/cms/fernanpe/Muminus_E-200To4000_Barrel-gun/Muminus_E-200To4000_Barrel-gun-PU/210124_185457/0000/ --fileNumber $1 --kind minus_barrel_pu #554
#python reader.py --inputDir /gpfs/projects/cms/fernanpe/Muminus_E-200To4000_Barrel-gun/Muminus_E-200To4000_Barrel-gun-NoPU/210122_202113/0000/ --fileNumber $1 --kind minus_barrel_nopu #97

#python reader.py --inputDir /gpfs/projects/cms/fernanpe/Muminus_E-200To4000_positiveOE-gun/Muminus_E-200To4000_positiveOE-gun-PU/210124_203228/0000/ --fileNumber $1 --kind minus_pos_pu # 293
#python reader.py --inputDir /gpfs/projects/cms/fernanpe/Muminus_E-200To4000_positiveOE-gun/Muminus_E-200To4000_positiveOE-gun-NoPU/210124_192237/0000/ --fileNumber $1 --kind minus_pos_nopu #49

#python reader.py --inputDir /gpfs/projects/cms/fernanpe/Muminus_E-200To4000_negativeOE-gun/Muminus_E-200To4000_negativeOE-gun-PU/210124_185618/0000/ --fileNumber $1 --kind minus_neg_pu #294
#python reader.py --inputDir /gpfs/projects/cms/fernanpe/Muminus_E-200To4000_negativeOE-gun/Muminus_E-200To4000_negativeOE-gun-NoPU/210124_185541/0000/ --fileNumber $1 --kind minus_neg_nopu #51

#python reader.py --inputDir /gpfs/projects/cms/fernanpe/Muplus_E-200To4000_Barrel-gun/Muplus_E-200To4000_Barrel-gun-PU/210124_204557/0000/ --fileNumber $1 --kind plus_barrel_pu #530
python reader.py --inputDir /gpfs/projects/cms/fernanpe/Muplus_E-200To4000_Barrel-gun/Muplus_E-200To4000_Barrel-gun-NoPU/210124_203814/0000/ --fileNumber $1 --kind plus_barrel_nopu #96

#python reader.py --inputDir /gpfs/projects/cms/fernanpe/Muplus_E-200To4000_positiveOE-gun/Muplus_E-200To4000_positiveOE-gun-PU/210124_205500/0000/ --fileNumber $1 --kind plus_pos_pu #284
#python reader.py --inputDir /gpfs/projects/cms/fernanpe/Muplus_E-200To4000_positiveOE-gun/Muplus_E-200To4000_positiveOE-gun-NoPU/210124_205352/0000/ --fileNumber $1 --kind plus_pos_nopu #51


#python reader.py --inputDir /gpfs/projects/cms/fernanpe/Muplus_E-200To4000_negativeOE-gun/Muplus_E-200To4000_negativeOE-gun-PU/210124_205319/0000/ --fileNumber $1 --kind plus_neg_pu #290
#python reader.py --inputDir /gpfs/projects/cms/fernanpe/Muplus_E-200To4000_negativeOE-gun/Muplus_E-200To4000_negativeOE-gun-NoPU/210124_205240/0000/ --fileNumber $1 --kind plus_neg_nopu #52
