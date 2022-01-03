import os
import shutil


for i in range(1,554):
    
    #print i

    subcom = 'sbatch -o logfile.log -e errfile.err --qos=cms_main --partition=cloudcms step1_CNN.sh ' + str(i)
    os.system(subcom)
