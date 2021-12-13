from CRABClient.UserUtilities import config
config = config()

config.section_("General")
config.General.transferLogs = True
config.General.requestName = 'Muminus_E-200To4000_positiveOE-gun-PU' 
config.General.workArea = 'crab_projects'


config.section_("JobType")
config.JobType.pluginName  = 'Analysis'
config.JobType.psetName = 'runRECOAnalysis_cfg.py'
#config.JobType.maxMemoryMB = 5000
#config.JobType.numCores = 8
config.JobType.outputFiles = ['tree.root']

config.section_("Data")
config.Data.splitting   = 'FileBased'
config.Data.unitsPerJob = 1

config.Data.outputDatasetTag = 'Muminus_E-200To4000_positiveOE-gun-PU'

config.Data.inputDBS = 'global'
config.Data.outLFNDirBase = '/store/user/fernanpe/' 
config.Data.publication = False
config.Data.outputDatasetTag = 'Muminus_E-200To4000_positiveOE-gun-PU'

config.Data.inputDataset = '/Muminus_E-200To4000_positiveOE-gun/Run3Winter20DRPremixMiniAOD-110X_mcRun3_2021_realistic_v6-v1/GEN-SIM-RECO'


config.section_("Site")
config.Site.storageSite = 'T2_ES_IFCA'
