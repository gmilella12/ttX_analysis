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
parser.add_argument('--files_per_job', dest='files_per_job', type=int, help='Number of files per job if grouping is enabled', default=1)
args = parser.parse_args()

if args.analyzer == 'test':
    print('!! WARNING: no python analyzer script selected !!')

file_list = []

NFS_PATH = os.environ.get('NFS', '/data/dust/user/gmilella')
ROOT_DIR = f'{NFS_PATH}/ttX_ntuplizer/'

if args.is_data:
    ROOT_DIR += 'data_'+args.year+'_hotvr/merged/'
elif args.is_sgn:
    ROOT_DIR += 'sgn_'+args.year+'_central_hotvr/merged/' #'
elif args.is_sgn_private:
    ROOT_DIR += 'sgn_'+args.year+'_hotvr/merged/'
else:
    ROOT_DIR += 'bkg_'+args.year+'_hotvr/merged/' #_single_muon'

for subdir, dirs, files in os.walk(ROOT_DIR):
    if 'log' in subdir:
        continue
    if 'topNN' in subdir:
        continue
    for file in files:
        if '.root' not in file: continue
        # if 'qcd_' in file: continue
        # if 'semilep' in file: continue
        # if '_300' not in file and '_500_' not in file and '_700' not in file: continue
        if 'nlo' in file: continue
        if 'dy_to' in file and args.year == '2018': continue
        if 'dy_to' in file and args.year == '2017': continue
        # if 'dy_to2L_m-50' not in file: continue
        # if 'ttW' not in file and 'ttZ' not in file and 'ttH' not in file: continue
        # if 'tt_semil' not in file: continue
        if 'Single' not in file: continue


        file_list.append(os.path.join(subdir, file))

print(f"Found {len(file_list)} files.")

command_str = ''
exe_args = []

if args.files_per_job != 1:
    grouped_files = [file_list[i:i + args.files_per_job] for i in range(0, len(file_list), args.files_per_job)]
else:
    grouped_files = [[f] for f in file_list]

print(f"Number of jobs to submit: {len(grouped_files)}")

with open('ttX_analysis_condor_submission', 'r') as condor_f:
    condor_sub_file = condor_f.read()
    condor_sub_file = condor_sub_file.replace(
        'EXE',
        'ttX_analysis_executable/'+args.year+'/ttX_analysis_executable_'+args.analyzer+'.sh'
    )

with open('ttX_analysis_condor_submission_new', 'w+') as condor_f_new:
    condor_f_new.write(condor_sub_file)

    for job_id, file_group in enumerate(grouped_files):
        input_files_str = " ".join(file_group)
        # print(input_files_str)

        command_str = f'arguments = "--input_files {input_files_str} --output_dir {args.output_dir} '

        if args.is_data:
            command_str += '--is_data '
        elif args.is_sgn or args.is_sgn_private:
            command_str += '--is_sgn '

        command_str += '"\n'
        condor_f_new.write(command_str)

        process_filename = os.path.basename(file_group[0])#.replace('.root', '')
        # --- extracting file name without .root
        match = re.search(r'([^/]+)\.root$', str(process_filename))
        if match: 
            process_filename = match.group(1)
        else:
            print("File name not extracted properly...")
            sys.exit()
        # ---

        condor_f_new.write(
            f'Output = {args.output_dir}/log/log_{args.analyzer}/log_{process_filename}_job{job_id}.$(Cluster).$(Process).out\n'
            f'Error = {args.output_dir}/log/log_{args.analyzer}/log_{process_filename}_job{job_id}.$(Cluster).$(Process).err\n'
            f'Log = {args.output_dir}/log/log_{args.analyzer}/log_{process_filename}_job{job_id}.$(Cluster).$(Process).log\nqueue\n'
        )

        # else: 
        #     print("WRONG PATH FILE or FILE DOES NOT EXIST: ", inFile)


if not os.path.isdir(args.output_dir):
    os.makedirs(args.output_dir)
LOG_REPO = '{}/log/log_{}'.format(args.output_dir, args.analyzer)
if not os.path.isdir(LOG_REPO):
    os.makedirs(LOG_REPO)

with open('ttX_analysis_executable_tmp.sh', 'r') as exe_f_tmp:
     exe_file = exe_f_tmp.read()
exe_file = exe_file.replace('ANALYZER', args.analyzer)
exe_file = exe_file.replace('YEAR', args.year)

if not os.path.exists('ttX_analysis_executable/{}'.format(args.year)):
    os.makedirs('ttX_analysis_executable/{}'.format(args.year))
exe_out = 'ttX_analysis_executable/{}/ttX_analysis_executable_{}.sh'.format(args.year, args.analyzer)
with open(exe_out, 'w') as exe_f:
    exe_f.write(exe_file)

    # for exe_cmd in exe_args:
    #     exe_f.write(
    #         'python {}.py {} --year {} \n'.format(args.analyzer, exe_cmd, args.year)
    #     )

os.chmod(exe_out, 0o777)

