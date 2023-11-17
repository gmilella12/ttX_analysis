import os
import subprocess
# from pathlib import Path
import argparse
parser = argparse.ArgumentParser()

parser.add_argument('--is_sgn', dest='is_sgn', action='store_true', help='Signal', default=False)
parser.add_argument('--year', dest='year', type=str, help='Year')
parser.add_argument('--is_data', dest='is_data', action='store_true', help='is_data', default=False)
parser.add_argument('--analyzer', dest='analyzer', type=str, help='.py analyzer', default='test')
parser.add_argument('--output_dir', dest='output_dir', type=str, required=True)
parser.add_argument('--weighting', dest='weighting', action='store_true')
args = parser.parse_args()

#subprocess.call("voms-proxy-init --rfc --voms cms -valid 192:00", shell = True)
#subprocess.call("echo $X509_USER_PROXY", shell = True)
#subprocess.call("export X509_USER_PROXY=/afs/desy.de/user/g/gmilella/.globus/x509", shell = True)

listOfDataSets = []
listOutputDir = []
yaml_file_dict = {}

file_list = []
process_list = []

import os

rootdir = '/nfs/dust/cms/user/gmilella/ttX_ntuplizer/'

if args.is_data:
    rootdir+='data_'+args.year+'/merged/'
elif args.is_sgn:
    rootdir+='sgn_'+args.year+'_hotvr/merged/'
else:
    #rootdir+='backup/bkg_'+args.year+'/'
    rootdir+='bkg_'+args.year+'_hotvr/merged/'

for subdir, dirs, files in os.walk(rootdir):
    if 'log' in subdir:
        continue
    if 'topNN' in subdir:
        continue

    for file in files:
        file_list.append(os.path.join(subdir, file))
                
           
print(file_list)

output_dir = os.getcwd() #'/afs/desy.de/user/g/gmilella/plotting_ttX_analysis'    

with open('ttX_analysis_condor_submission', 'r') as condor_f:
    condor_sub_file = condor_f.read()
    condor_sub_file = condor_sub_file.replace('EXE','ttX_analysis_executable/'+args.year+'/ttX_analysis_executable_'+args.analyzer+'.sh')

with open('ttX_analysis_condor_submission_new', 'w+') as condor_f_new: 
    condor_f_new.write(condor_sub_file)    

    for i, inFile in enumerate(file_list):
        #print(inFile)

        if 'ttX_mass' in inFile: 
            process_file = inFile[:-5].rsplit("/",1)[1]
            process = process_file.rsplit("_",2)[0]
            #print(process_file)
            #print(process)
        elif 'data' in inFile:
            process_file = inFile[:-5].rsplit("/",1)[1]
            #print(process_file)
            process = process_file.rsplit("_",3)[0]
            #print(process_file)
            #print(process)
        else:
            process_file = inFile[:-5].rsplit("/",1)[1]
            process = process_file.rsplit("_",4)[0]
            #print(process_file)
            #print(process)

        #print(process)
        if os.path.exists(inFile):
            if args.is_data:
                if args.weighting: condor_f_new.write('arguments = "--input_file '+inFile+' --is_data --output_dir '+args.output_dir+' --weighting"\n')
                else: condor_f_new.write('arguments = "--input_file '+inFile+' --is_data --output_dir '+args.output_dir+' "\n')
            elif args.is_sgn: 
                if args.weighting: condor_f_new.write('arguments = "--input_file '+inFile+' --is_sgn --output_dir '+args.output_dir+' --weighting"\n')
                else: condor_f_new.write('arguments = "--input_file '+inFile+' --is_sgn --output_dir '+args.output_dir+' "\n')
            else: 
                if args.weighting: condor_f_new.write('arguments = "--input_file '+inFile+' --output_dir '+args.output_dir+' --weighting"\n')
                else: condor_f_new.write('arguments = "--input_file '+inFile+' --output_dir '+args.output_dir+' "\n')
        else: 
            print("WRONG PATH FILE or FILE DOES NOT EXIST: ", inFile)

        condor_f_new.write('Output = '+output_dir+'/log/log_'+args.analyzer+'/log_'+process_file+'.$(Process).out\nError = '+output_dir+'/log/log_'+args.analyzer+'/log_'+process_file+'.$(Process).err\nLog = '+output_dir+'/log/log_'+args.analyzer+'/log_'+process_file+'.$(Process).log\nqueue\n')
            
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
    if not os.path.isdir(output_dir+'/log/log_'+args.analyzer):
        os.makedirs(output_dir+'/log/log_'+args.analyzer)


with open('ttX_analysis_executable_tmp.sh', 'r') as exe_f_tmp:
     exe_file = exe_f_tmp.read()
exe_file=exe_file.replace('ANALYZER', args.analyzer)
exe_file=exe_file.replace('YEAR', args.year)

with open('ttX_analysis_executable/'+args.year+'/ttX_analysis_executable_'+args.analyzer+'.sh', 'w') as exe_f:
     exe_f.write(exe_file)
os.chmod('ttX_analysis_executable/'+args.year+'/ttX_analysis_executable_'+args.analyzer+'.sh', 0o777)

