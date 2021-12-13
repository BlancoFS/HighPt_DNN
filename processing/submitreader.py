import os
import shutil

#subcom = 'sbatch -o logfile.log -e errfile.err --qos=cms_med --partition=cloudcms reader.sh ' + str(18)
#os.system(subcom)


for i in range(1,530): #(i,j-1)
   subcom = 'sbatch -o logfile.log -e errfile.err --qos=cms_med --partition=cloudcms reader.sh ' + str(i)
   os.system(subcom)

