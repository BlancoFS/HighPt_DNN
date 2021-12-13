Train (grid):

      python submit.py

Evaluate the trained models on the test dataset:

      python submit_eval.py

Get the final plots:

    sbatch -o logfile.log -e errfile.err --qos=cms_main --partition=cloudcms --mem-per-cpu=30G submit_plots.sh