import re, sys, os

import ROOT

import yaml
from yaml.loader import SafeLoader

ROOT_DIR = os.getcwd()

def parsing_file(input_file):
    print("Processing file: {}".format(input_file))
    pattern = re.compile(r"\/([^/]*?)_MC")
    match = pattern.search(str(input_file))
    if match:
        print("Process name: {}".format(match.group(1)))
        return match.group(1)
    else:
        pattern = re.compile(r"\/([^/]*?)_ntuplizer")
        match = pattern.search(str(input_file))
        if match:
            print("Process name: {}".format(match.group(1)))
            return match.group(1)
        else:
            pattern = re.compile(r"\/([^/]*?)(?:_\d+)?_output")
            match = pattern.search(str(input_file))
            if match:
                print("Process name: {}".format(match.group(1)))
                return match.group(1)
            else:
                pattern = re.compile(r"\/([^/]*?)_merged")
                match = pattern.search(str(input_file))
                if match:
                    print("Process name: {}".format(match.group(1)))
                    return match.group(1)
                else:
                    print("No process name found.")
                    sys.exit()

def xsec(process_name, is_sgn):
    if is_sgn:
        pattern = r"_width\d+"
        process_name = re.sub(pattern, '', process_name, flags=re.I)

    with open('{}/xsec.yaml'.format(ROOT_DIR)) as xsec_file:
        xsecFile = yaml.load(xsec_file, Loader=SafeLoader)
    if xsecFile[process_name]['isUsed']:
        return xsecFile[process_name]['xSec']
    else:
        print("Xsec for process {} not found in file".format(process_name))
        sys.exit()

def sum_gen_weights(input_file, process_name, is_sgn, year):
    if is_sgn:
        root_file = ROOT.TFile(str(input_file), 'READ')
        sumgenweight = root_file.Get("sumGenWeights")
        return sumgenweight.GetVal()
    else:
        with open("{}/test_files/sum_gen_weights.yaml".format(ROOT_DIR)) as sumGenWeights_file:
            sumGenWeightsFile = yaml.load(sumGenWeights_file, Loader=SafeLoader)
            return sumGenWeightsFile[process_name]

def creation_output_file(input_file, output_dir, analyzer, year, event_selection, systematic):
    # bkg samples are divided in chunks
    # so the output files should reflect this division
    match = re.search(r'([^/]+)\.root$', str(input_file))
    if match: 
        process_filename = match.group(1)
    else:
        print("File name not extracted properly...")
        sys.exit()

    # check if dir exist
    output_dir_path = os.path.join(output_dir, str(year), analyzer, systematic)
    # output_dir_path = os.path.join(output_dir, str(year), "{}_{}".format(analyzer, event_selection), systematic)
    if not os.path.exists(output_dir_path):
        os.makedirs(output_dir_path)

    output_path = os.path.join(output_dir_path, "{}_{}_{}_{}.root".format(process_filename, analyzer, event_selection, systematic))
    # output_path = os.path.join(output_dir_path, "{}_{}_{}.root".format(process_filename, analyzer, systematic))

    file_out = ROOT.TFile(output_path, 'RECREATE')
    for lep_sel in ['emu', 'ee', 'mumu']:
        ROOT.gDirectory.mkdir(lep_sel)
        file_out.cd()
    print("Output file {}: ".format(file_out))
    return file_out

def creation_output_h5_file(input_file, output_dir, analyzer, year, event_selection, systematic):
    # bkg samples are divided in chunks
    # so the output files should reflect this division
    match = re.search(r'([^/]+)\.root$', str(input_file))
    if match: 
        process_filename = match.group(1)
    else:
        print("File name not extracted properly...")
        sys.exit()

    # check if dir exist
    output_dir_path = os.path.join(output_dir, str(year), analyzer, systematic)
    # output_dir_path = os.path.join(output_dir, str(year), "{}_{}".format(analyzer, event_selection), systematic)
    if not os.path.exists(output_dir_path):
        os.makedirs(output_dir_path)

    # output_path = os.path.join(output_dir_path, "{}_{}_{}_{}.root".format(process_filename, analyzer, event_selection, systematic))
    output_path = os.path.join(output_dir_path, "{}_{}_{}.h5".format(process_filename, analyzer, systematic))

    print("Output file {}: ".format(output_path))
    return output_path