import os
import shutil
import glob


for myfile in glob.glob("send_doEval*sh"):

    subcom = 'sbatch -o logfile.log -e errfile.err --qos=cms_main --partition=cloudcms ' + myfile
    os.system(subcom)


 
 


                    


