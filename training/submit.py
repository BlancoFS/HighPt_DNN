import os
import shutil
import glob


for myfile in glob.glob("send_nFirst*sh"):

    subcom = 'sbatch -o logfile.log -e errfile.err --qos=cms_main --partition=cloudcms --mem-per-cpu=14G ' + myfile
    os.system(subcom)


 
 


                    


