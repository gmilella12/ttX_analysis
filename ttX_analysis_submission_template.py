import os, sys
import re
import argparse
parser = argparse.ArgumentParser()

parser.add_argument('--is_sgn', dest='is_sgn', action='store_true', help='Signal', default=False)
parser.add_argument('--is_sgn_private', dest='is_sgn_private', action='store_true', help='Signal private production', default=False)
parser.add_argument('--year', dest='year', type=str, help='Year')
parser.add_argument('--is_data', dest='is_data', action='store_true', help='is_data', default=False)
parser.add_argument('--analyzer', dest='analyzer', type=str, help='.py analyzer', default='test')
parser.add_argument('--output_dir', dest='output_dir', type=str, required=True)
parser.add_argument('--weighting', dest='weighting', action='store_true', default=False)
args = parser.parse_args()

listOfDataSets = []
listOutputDir = []
yaml_file_dict = {}

file_list = []
process_list = []

ROOT_DIR = '/nfs/dust/cms/user/gmilella/ttX_ntuplizer/'

if args.is_data:
    ROOT_DIR += 'data_'+args.year+'_hotvr/merged/'
elif args.is_sgn:
    ROOT_DIR += 'sgn_'+args.year+'_central_hotvr/merged/'
elif args.is_sgn_private:
    ROOT_DIR += 'sgn_'+args.year+'_hotvr/merged/'
else:
    ROOT_DIR += 'bkg_'+args.year+'_hotvr/merged/' 

for subdir, dirs, files in os.walk(ROOT_DIR):
    if 'log' in subdir:
        continue
    if 'topNN' in subdir:
        continue

    for file in files:
        file_list.append(os.path.join(subdir, file))

print(file_list)

batch_number = 1 
batch_count = 0
command_str = ''
command_dict = dict()

with open('ttX_analysis_condor_submission', 'r') as condor_f:
    condor_sub_file = condor_f.read()
    condor_sub_file = condor_sub_file.replace('EXE','ttX_analysis_executable/'+args.year+'/ttX_analysis_executable_'+args.analyzer+'.sh')

with open('ttX_analysis_condor_submission_new', 'w+') as condor_f_new: 
    condor_f_new.write(condor_sub_file)    

    for i, inFile in enumerate(file_list):
        if '.root' not in inFile: continue
        # if 'dy_m-50boosted_jets' not in inFile: continue
        
        # if '_F' not in inFile: continue

        # --- extracting file name without .root
        match = re.search(r'([^/]+)\.root$', str(inFile))
        if match: 
            process_filename = match.group(1)
        else:
            print("File name not extracted properly...")
            sys.exit()
        # ---

        #print(process)
        if os.path.exists(inFile):
            command_str = 'arguments = "--input_file {} --output_dir {} '.format(inFile, args.output_dir)
            if args.is_data:
                command_str += ' --is_data '
            elif args.is_sgn or args.is_sgn_private: 
                command_str += ' --is_sgn '

            if args.weighting: 
                command_str += ' --weighting '
            
            command_str += '"\n'
            condor_f_new.write(command_str)
        else: 
            print("WRONG PATH FILE or FILE DOES NOT EXIST: ", inFile)

        condor_f_new.write(
            'Output = {}/log/log_{}/log_{}.$(Process).out\n'
            'Error = {}/log/log_{}/log_{}.$(Process).err\n'
            'Log = {}/log/log_{}/log_{}.$(Process).log\nqueue\n'
            .format(args.output_dir, args.analyzer, process_filename,
                    args.output_dir, args.analyzer, process_filename,
                    args.output_dir, args.analyzer, process_filename)
        )

if not os.path.isdir(args.output_dir):
    os.makedirs(args.output_dir)
LOG_REPO = '{}/log/log_{}'.format(args.output_dir, args.analyzer)
if not os.path.isdir(LOG_REPO):
    os.makedirs(LOG_REPO)

with open('ttX_analysis_executable_tmp.sh', 'r') as exe_f_tmp:
     exe_file = exe_f_tmp.read()
exe_file = exe_file.replace('ANALYZER', args.analyzer)
exe_file = exe_file.replace('YEAR', args.year)

exe_out = 'ttX_analysis_executable/{}/ttX_analysis_executable_{}.sh'.format(args.year, args.analyzer)
with open(exe_out, 'w') as exe_f:
     exe_f.write(exe_file)
os.chmod(exe_out, 0o777)

