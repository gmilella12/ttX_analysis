import re, sys, os
import csv
import ROOT

import yaml
from yaml.loader import SafeLoader

sys.path.append(os.getcwd())
from utils_folder.utils import *

import ROOT
ROOT.ROOT.EnableImplicitMT()

ROOT_DIR = os.getcwd()

headers = {
    "cpp_functions_header": f"{ROOT_DIR}/cpp_functions_header.h",
    "cpp_top_mistagging_scale_factor_header": f"{ROOT_DIR}/cpp_top_mistagging_scale_factor_header.h",
    "cpp_trigger_scale_factor_header": f"{ROOT_DIR}/cpp_trigger_scale_factor_header.h",
    "cpp_puID_scale_factor_header": f"{ROOT_DIR}/cpp_puID_scale_factor_header.h",
    "cpp_mc_scaling_header": f"{ROOT_DIR}/cpp_mc_scaling_header.h"
}

for name, path in headers.items():
    if not os.path.isfile(path):
        print(f"Error: Header file '{path}' not found!")
        sys.exit()
    else:
        print(f"Found header file: {path}")
        ROOT.gInterpreter.Declare(f'#include "{path}"')

NFS_PATH = os.environ.get('NFS', '/data/dust/user/gmilella')

B_TAGGING_WP = {
    '2016preVFP': 
        {'loose': 0.0508, 'medium': 0.2598, 'tight': 0.6502}, #https://btv-wiki.docs.cern.ch/ScaleFactors/UL2016preVFP/
    '2016': 
        {'loose': 0.0480, 'medium': 0.2489, 'tight': 0.6377}, #https://btv-wiki.docs.cern.ch/ScaleFactors/UL2016postVFP/
    '2017': 
        {'loose': 0.0532, 'medium': 0.3040, 'tight': 0.7476}, #https://btv-wiki.docs.cern.ch/ScaleFactors/UL2017/
    '2018': 
        {'loose': 0.0490, 'medium': 0.2783, 'tight': 0.7100}, #https://btv-wiki.docs.cern.ch/ScaleFactors/UL2018/
    '2022': 
        {'loose': 0.0583, 'medium': 0.3086, 'tight': 0.7183}, #https://btv-wiki.docs.cern.ch/ScaleFactors/Run3Summer22/
    '2022EE': 
        {'loose': 0.0614, 'medium': 0.3196, 'tight': 0.73}, #https://btv-wiki.docs.cern.ch/ScaleFactors/Run3Summer22EE/
}

# TRIGGER_SEL = {
#     '2016preVFP': 
#         {'ee': 'trigger_HLT_Ele23_Ele12_CaloIdL_TrackIdL_IsoVL_DZ', 
#          'mumu': '(trigger_HLT_Mu17_TrkIsoVVL_Mu8_TrkIsoVVL || trigger_HLT_Mu17_TrkIsoVVL_TkMu8_TrkIsoVVL)', 
#          'emu': ' (trigger_HLT_Mu23_TrkIsoVVL_Ele12_CaloIdL_TrackIdL_IsoVL || trigger_HLT_Mu8_TrkIsoVVL_Ele23_CaloIdL_TrackIdL_IsoVL) ',
#          'single_muon': 'trigger_HLT_IsoMu24', #'(trigger_HLT_IsoMu24 || trigger_HLT_IsoTkMu24)'
#          'or': '(trigger_ee_flag || trigger_emu_flag || trigger_mumu_flag)'
#          }, 
#     '2016': 
#         {'ee': 'trigger_HLT_Ele23_Ele12_CaloIdL_TrackIdL_IsoVL_DZ', 
#          'mumu': '(trigger_HLT_Mu17_TrkIsoVVL_Mu8_TrkIsoVVL || trigger_HLT_Mu17_TrkIsoVVL_TkMu8_TrkIsoVVL)', 
#          'emu': ' (trigger_HLT_Mu23_TrkIsoVVL_Ele12_CaloIdL_TrackIdL_IsoVL || trigger_HLT_Mu8_TrkIsoVVL_Ele23_CaloIdL_TrackIdL_IsoVL)',
#          'single_muon': 'trigger_HLT_IsoMu24', #'(trigger_HLT_IsoMu24 || trigger_HLT_IsoTkMu24)'
#          'or': '(trigger_ee_flag || trigger_emu_flag || trigger_mumu_flag)'
#          },
#     '2017': 
#         {'ee': 'trigger_HLT_Ele23_Ele12_CaloIdL_TrackIdL_IsoVL', 
#          'mumu': ' trigger_HLT_Mu17_TrkIsoVVL_Mu8_TrkIsoVVL_DZ_Mass8', 
#          'emu': '(trigger_HLT_Mu23_TrkIsoVVL_Ele12_CaloIdL_TrackIdL_IsoVL || trigger_HLT_Mu8_TrkIsoVVL_Ele23_CaloIdL_TrackIdL_IsoVL_DZ)',
#          'single_muon': 'trigger_HLT_IsoMu27',
#         'or': '(trigger_ee_flag || trigger_emu_flag || trigger_mumu_flag)'
#          },
#     '2018': 
#         {'ee': 'trigger_HLT_Ele23_Ele12_CaloIdL_TrackIdL_IsoVL', 
#          'mumu': 'trigger_HLT_Mu17_TrkIsoVVL_Mu8_TrkIsoVVL_DZ_Mass3p8', 
#          'emu': '(trigger_HLT_Mu23_TrkIsoVVL_Ele12_CaloIdL_TrackIdL_IsoVL || trigger_HLT_Mu8_TrkIsoVVL_Ele23_CaloIdL_TrackIdL_IsoVL_DZ)',
#          'single_muon': 'trigger_HLT_IsoMu24',
#          'or': '(trigger_ee_flag || trigger_emu_flag || trigger_mumu_flag)'
#          }, 
#     '2022': 
#         {'ee':'(trigger_HLT_Ele23_Ele12_CaloIdL_TrackIdL_IsoVL_DZ || trigger_HLT_Ele23_Ele12_CaloIdL_TrackIdL_IsoVL)', 
#          'mumu': 'trigger_HLT_Mu17_TrkIsoVVL_Mu8_TrkIsoVVL_DZ_Mass3p8', 
#          'emu': '(trigger_HLT_Mu23_TrkIsoVVL_Ele12_CaloIdL_TrackIdL_IsoVL || trigger_HLT_Mu8_TrkIsoVVL_Ele23_CaloIdL_TrackIdL_IsoVL_DZ)',
#          'single_muon': 'trigger_HLT_IsoMu24',
#          'or': '(trigger_ee_flag || trigger_emu_flag || trigger_mumu_flag)'
#          }, 
#     '2022EE': 
#         {'ee':'(trigger_HLT_Ele23_Ele12_CaloIdL_TrackIdL_IsoVL_DZ || trigger_HLT_Ele23_Ele12_CaloIdL_TrackIdL_IsoVL)', 
#          'mumu': 'trigger_HLT_Mu17_TrkIsoVVL_Mu8_TrkIsoVVL_DZ_Mass3p8', 
#          'emu': '(trigger_HLT_Mu23_TrkIsoVVL_Ele12_CaloIdL_TrackIdL_IsoVL || trigger_HLT_Mu8_TrkIsoVVL_Ele23_CaloIdL_TrackIdL_IsoVL_DZ)',
#          'single_muon': 'trigger_HLT_IsoMu24',
#          'or': '(trigger_ee_flag || trigger_emu_flag || trigger_mumu_flag)'
#          }, 
# }

JET_COMPOSITION_FLAGS = ['has_gluon_or_quark_not_fromTop']
JET_COMPOSITION_FLAGS.extend(['has_other','has_b_not_fromTop'])
for flag_top_inside in ['topIsInside','topIsNotInside','topIsNotInside_and_has_gluon_or_quark_not_fromTop']:
    JET_COMPOSITION_FLAGS.append('has_hadronicTop_'+flag_top_inside)
    JET_COMPOSITION_FLAGS.append('has_other_'+flag_top_inside)
    JET_COMPOSITION_FLAGS.append('has_noTopDaughters_'+flag_top_inside)
    if flag_top_inside == 'topIsInside' or flag_top_inside == 'topIsNotInside_and_has_gluon_or_quark_not_fromTop':
        for top_label in ['fromTop']: #,'not_fromTop']:
            JET_COMPOSITION_FLAGS.append('has_leptonicW_'+top_label+'_'+flag_top_inside)
            JET_COMPOSITION_FLAGS.append('has_hadronicW_'+top_label+'_'+flag_top_inside)
            JET_COMPOSITION_FLAGS.append('has_b_plus_quark_'+top_label+'_'+flag_top_inside)
            JET_COMPOSITION_FLAGS.append('has_b_plus_lepton_'+top_label+'_'+flag_top_inside)
            JET_COMPOSITION_FLAGS.append('has_b_'+top_label+'_'+flag_top_inside)
            JET_COMPOSITION_FLAGS.append('has_quark_fromW_'+top_label+'_'+flag_top_inside)
# --- special case, samples without top and samples like tW, ttH
JET_COMPOSITION_FLAGS.append('has_leptonicW_not_fromTop')
JET_COMPOSITION_FLAGS.append('has_hadronicW_not_fromTop')
JET_COMPOSITION_FLAGS.append('has_b_plus_quark_not_fromTop')
JET_COMPOSITION_FLAGS.append('has_b_plus_lepton_not_fromTop')
JET_COMPOSITION_FLAGS.append('has_b_not_fromTop')
JET_COMPOSITION_FLAGS.append('has_quark_fromW_not_fromTop')

seen = set()
JET_COMPOSITION_FLAGS_UNIQUE = [x for x in JET_COMPOSITION_FLAGS if not (x in seen or seen.add(x))]

HADRONIC_W_FLAG, B_FROM_TOP_FLAG, B_NOT_FROM_TOP_FLAG = '', '', ''
Q_FROM_W_PLUS_B_FLAG, Q_FROM_W_FLAG, HADRONIC_T_FLAG = '', '', ''
OTHERS = ''
# --- tW process flags
HADRONIC_W_NOT_FROM_T_FLAG = ''
Q_FROM_W_NOT_FROM_T_FLAG = ''
# ---
PURE_QCD_FLAGS = ''
ALL_FLAGS = ''

for jet_flag in JET_COMPOSITION_FLAGS_UNIQUE:
    ALL_FLAGS += 'selectedHOTVRJets_nominal_'+jet_flag + '||'
    if jet_flag == 'has_hadronicW_fromTop_topIsInside':
        HADRONIC_W_FLAG = 'selectedHOTVRJets_nominal_'+jet_flag
    elif jet_flag == 'has_b_fromTop_topIsInside':
        B_FROM_TOP_FLAG = 'selectedHOTVRJets_nominal_'+jet_flag
    elif jet_flag == 'has_b_not_fromTop':
        B_NOT_FROM_TOP_FLAG = 'selectedHOTVRJets_nominal_'+jet_flag
    elif jet_flag == 'has_b_plus_quark_fromTop_topIsInside':
        Q_FROM_W_PLUS_B_FLAG = 'selectedHOTVRJets_nominal_'+jet_flag 
    elif jet_flag == 'has_quark_fromW_fromTop_topIsInside':
        Q_FROM_W_FLAG = 'selectedHOTVRJets_nominal_'+jet_flag
    elif jet_flag == 'has_hadronicTop_topIsInside':
        HADRONIC_T_FLAG = 'selectedHOTVRJets_nominal_'+jet_flag
    elif jet_flag == 'has_hadronicW_not_fromTop':
        HADRONIC_W_NOT_FROM_T_FLAG = 'selectedHOTVRJets_nominal_'+jet_flag
    elif jet_flag == 'has_quark_fromW_not_fromTop':
        Q_FROM_W_NOT_FROM_T_FLAG = 'selectedHOTVRJets_nominal_'+jet_flag
    elif jet_flag == 'has_leptonicW_fromTop_topIsInside' or jet_flag == 'has_hadronicTop_topIsNotInside' or jet_flag == 'has_b_plus_lepton_fromTop_topIsInside' or jet_flag == 'has_leptonicW_not_fromTop' or jet_flag == 'has_b_plus_quark_not_fromTop' or jet_flag == 'has_b_plus_lepton_not_fromTop':
        OTHERS += 'selectedHOTVRJets_nominal_'+jet_flag + '||'
    else:
        PURE_QCD_FLAGS += 'selectedHOTVRJets_nominal_'+jet_flag + '||'


JET_COMPOSITION_FLAGS_MERGED = {
        'pure_qcd': PURE_QCD_FLAGS[:-2], 
        'hadronic_t': HADRONIC_T_FLAG,
        'hadronic_w': HADRONIC_W_FLAG, 
        'b_from_top': B_FROM_TOP_FLAG, 
        'b_not_from_top': B_NOT_FROM_TOP_FLAG,
        'q_from_w_plus_b': Q_FROM_W_PLUS_B_FLAG, 
        'q_from_w': Q_FROM_W_FLAG, 
        'others': OTHERS[:-2],
        'hadronic_w_not_from_t': HADRONIC_W_NOT_FROM_T_FLAG,
        'q_from_w_not_from_top': Q_FROM_W_NOT_FROM_T_FLAG,
        'non_covered': ALL_FLAGS[:-2],
}
Z_PEAK_LOW_EDGE, Z_PEAK_HIGH_EDGE = 80., 101.


def parsing_file(input_file):
    print("Processing file: {}".format(input_file))
    input_file = input_file.rsplit("/", 2)[-1]

    # Match ending in _MC
    pattern = re.compile(r"([^/]*?)_MC")
    match = pattern.search(str(input_file))
    if match:
        # print("Process name: {}".format(match.group(1)))
        return match.group(1)

    # Match ending in _ntuplizer
    pattern = re.compile(r"([^/]*?)_ntuplizer")
    match = pattern.search(str(input_file))
    if match:
        # print("Process name: {}".format(match.group(1)))
        return match.group(1)

    # Match ending in _output or _123_output
    pattern = re.compile(r"([^/]*?)(?:_\d+)?_output")
    match = pattern.search(str(input_file))
    if match:
        # print("Process name: {}".format(match.group(1)))
        return match.group(1)

    # Match pattern like TuneCP5
    pattern = re.compile(r"([^/]*?)_TuneCP5")
    match = pattern.search(str(input_file))
    if match:
        # print("Process name: {}".format(match.group(1)))
        return match.group(1)

    # Match ending in _<number>_merged.root
    pattern = r"(_[0-9]+_merged\.root)$"
    name = re.sub(pattern, '', str(input_file))
    if name != str(input_file):
        # print("Process name: {}".format(name))
        return name

    # Match ending in _merged.root
    pattern = r"(_merged\.root)$"
    name = re.sub(pattern, '', str(input_file))
    if name != str(input_file):
        # print("Process name: {}".format(name))
        return name

    print("No process name found.")
    sys.exit()

def xsec(process_name, is_sgn, year='2018'):
    if is_sgn:
        process_name = re.sub(r"_Width.*", '', process_name, flags=re.I)
        process_name = re.sub(r"_w.*", '', process_name, flags=re.I)

    if year == '2022' or year == '2022EE':
        with open('{}/xsec_Run3.yaml'.format(ROOT_DIR)) as xsec_file:
            xsecFile = yaml.load(xsec_file, Loader=SafeLoader)
    else:
        with open('{}/xsec_Run2.yaml'.format(ROOT_DIR)) as xsec_file:
            xsecFile = yaml.load(xsec_file, Loader=SafeLoader)

    if xsecFile[process_name]['isUsed']:
        return xsecFile[process_name]['xSec']
    else:
        print("Xsec for process {} not found in file".format(process_name))
        sys.exit()

def sum_gen_weights(input_file, process_name, is_sgn, year):
    if is_sgn:
        with open(f"{NFS_PATH}/ttX_ntuplizer/sgn_{year}_central_hotvr/merged/sum_gen_weights.yaml") as sumGenWeights_file:
            sumGenWeightsFile = yaml.load(sumGenWeights_file, Loader=SafeLoader)
            return sumGenWeightsFile[process_name]
        # with open(f"{NFS_PATH}/ttX_ntuplizer/sgn_{}_hotvr/merged/sum_gen_weights.yaml") as sumGenWeights_file:
        #     sumGenWeightsFile = yaml.load(sumGenWeights_file, Loader=SafeLoader)
        #     return sumGenWeightsFile[process_name]
    else:
        with open(f"{NFS_PATH}/ttX_ntuplizer/bkg_{year}_hotvr/merged/sum_gen_weights.yaml") as sumGenWeights_file:
            sumGenWeightsFile = yaml.load(sumGenWeights_file, Loader=SafeLoader)
            return sumGenWeightsFile[process_name]
        
def sum_lhe_scale_weights(input_file, process_name, is_sgn, year):
    if is_sgn:
        with open(f"{NFS_PATH}/ttX_ntuplizer/sgn_{year}_central_hotvr/merged/sum_lhe_scale_weights.yaml") as sumGenWeights_file:
            sumGenWeightsFile = yaml.load(sumGenWeights_file, Loader=SafeLoader)
            return sumGenWeightsFile[process_name]
        # with open(f"{NFS_PATH}/ttX_ntuplizer/sgn_{}_hotvr/merged/sum_lhe_scale_weights.yaml") as sumGenWeights_file:
        #     sumGenWeightsFile = yaml.load(sumGenWeights_file, Loader=SafeLoader)
        #     return sumGenWeightsFile[process_name]
    else:
        with open(f"{NFS_PATH}/ttX_ntuplizer/bkg_{year}_hotvr/merged/sum_lhe_scale_weights.yaml") as sumGenWeights_file:
            sumGenWeightsFile = yaml.load(sumGenWeights_file, Loader=SafeLoader)
            return sumGenWeightsFile[process_name]

def Z_peak_fit_results(year):
    # calculating with another script in the postprocessing (geometric mean between ee and mumu channel)
    csv_file_path = f"DY_normalization_offset/{year}/"

    with open(f"{csv_file_path}/Z_peak_fit_results.csv", mode="r") as csv_file:
        reader = csv.reader(csv_file)
        header = next(reader)
        values = next(reader)
    return values

def creation_output_file(input_file, output_dir, analyzer, year, event_selection, lepton_selection, systematic, miscellanea=''):
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

    output_path = os.path.join(
        output_dir_path, 
        f"{process_filename}_{analyzer}_{event_selection}_{systematic}{miscellanea}.root"
    )

    file_out = ROOT.TFile(output_path, 'RECREATE')
    systematic_dir = file_out.mkdir(systematic)
    for lep_sel in lepton_selection:
        lep_sel_path = f"{systematic}/{lep_sel}"
        file_out.mkdir(lep_sel_path)
    print("Output file {}: ".format(file_out))

    return file_out, output_path

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

def histo_creation(
        root_df, 
        histo_title, 
        histo_var, 
        binning={'bins_x': [], 'bins_x_edges': [], 'bins_y': [], 'bins_y_edges': []}, 
        is_data=False, 
        histo_type='TH1', 
        weight='',
        additional_filter=''
    ):

    if additional_filter:
        root_df = root_df.Filter(additional_filter, additional_filter)

    if histo_type == 'TH1':
        if is_data:
            if weight != None and 'tagging_efficiency' in weight:
                output_histo = root_df.Histo1D(
                    (histo_title, '', binning['bins_x'], binning['bins_x_edges']), 
                    histo_var['var_x'], weight
                )
            else:
                output_histo = root_df.Histo1D(
                    (histo_title, '', binning['bins_x'], binning['bins_x_edges']), 
                    histo_var['var_x']
                )
        else:
            if weight==None:
                output_histo = root_df.Histo1D(
                    (histo_title, '', binning['bins_x'], binning['bins_x_edges']), 
                    histo_var['var_x']
                )
            else:
                output_histo = root_df.Histo1D(
                    (histo_title, '', binning['bins_x'], binning['bins_x_edges']), 
                    histo_var['var_x'], weight
                )
    else:
        if is_data:
            if weight != None and 'tagging_efficiency' in weight:
                output_histo = root_df.Histo2D(
                    (histo_title, '', binning['bins_x'], binning['bins_x_edges'], binning['bins_y'], binning['bins_y_edges']), 
                    histo_var['var_x'], 
                    histo_var['var_y'], 
                    weight
                )
            else: 
                output_histo = root_df.Histo2D(
                    (histo_title, '', binning['bins_x'], binning['bins_x_edges'], binning['bins_y'], binning['bins_y_edges']), 
                    histo_var['var_x'], 
                    histo_var['var_y'], 
                )
        else:
            if weight == None: 
                output_histo = root_df.Histo2D(
                    (histo_title, '', binning['bins_x'], binning['bins_x_edges'], binning['bins_y'], binning['bins_y_edges']), 
                    histo_var['var_x'], 
                    histo_var['var_y'], 
                )
            else:
                output_histo = root_df.Histo2D(
                    (histo_title, '', binning['bins_x'], binning['bins_x_edges'], binning['bins_y'], binning['bins_y_edges']), 
                    histo_var['var_x'], 
                    histo_var['var_y'], 
                    weight
                )

    return output_histo


def make_cutflow_histograms(
    root_df, 
    filter_conditions, 
    weight_column="weight", 
    cutflow_name="cutflow", 
    use_weights=True
):
    cutflow_unweighted = []
    cutflow_weighted = []

    current_df = root_df
    for i, condition in enumerate(filter_conditions):
        label = f"Cut_{i+1}"
        current_df = current_df.Filter(condition, label)

        # Count unweighted
        n_unweighted = current_df.Count()
        cutflow_unweighted.append((label, n_unweighted))

        # Count weighted if applicable
        if use_weights:
            n_weighted = current_df.Define("w", weight_column).Sum("w")
            cutflow_weighted.append((label, n_weighted))

    if use_weights:
        return cutflow_unweighted, cutflow_weighted
    else:
        return cutflow_unweighted, None

def event_selection(
        root_df, 
        lepton_selection='ee', 
        n_leptons=2, 
        on_Z_peak='on',
        LEPTON_ID_ELE='medium', 
        ELECTRON_ID_TYPE='MVA', 
        LEPTON_ID_MUON='loose',
        n_ak4_outside=2, 
        n_b_outside=2, 
        b_wp='loose', 
        boosted_jets='hotvr', 
        ak4_systematic='nominal', 
        boosted_systematic='nominal',
        year='2018', 
        process='DoubleMuon', 
        return_cutflow=False,
        weight_column='',
        is_data=False
    ):

    TRIGGER_SEL = {
        '2016preVFP': 
            {
                'DoubleMuon': '(trigger_HLT_Mu17_TrkIsoVVL_Mu8_TrkIsoVVL || trigger_HLT_Mu17_TrkIsoVVL_TkMu8_TrkIsoVVL)',
                'SingleMuon': '(trigger_HLT_IsoMu24 || trigger_HLT_IsoTkMu24) && !(trigger_HLT_Mu17_TrkIsoVVL_Mu8_TrkIsoVVL || trigger_HLT_Mu17_TrkIsoVVL_TkMu8_TrkIsoVVL || trigger_emu_flag)',
                'SingleElectron': 'trigger_HLT_Ele32_WPTight_Gsf && !(trigger_HLT_Ele23_Ele12_CaloIdL_TrackIdL_IsoVL_DZ || trigger_emu_flag)',
                'DoubleLepton': '(trigger_HLT_Mu23_TrkIsoVVL_Ele12_CaloIdL_TrackIdL_IsoVL || trigger_HLT_Mu8_TrkIsoVVL_Ele23_CaloIdL_TrackIdL_IsoVL)',
                'DoubleEG': 'trigger_HLT_Ele23_Ele12_CaloIdL_TrackIdL_IsoVL_DZ',
                'ee': 'trigger_ee_flag',
                'emu': 'trigger_emu_flag',
                'mumu': 'trigger_mumu_flag',
                'single_muon': 'trigger_HLT_IsoMu24'
            }, 
        '2016': 
            {
                'DoubleMuon': '(trigger_HLT_Mu17_TrkIsoVVL_Mu8_TrkIsoVVL || trigger_HLT_Mu17_TrkIsoVVL_TkMu8_TrkIsoVVL)',
                'DoubleMuon_2016_H': '(trigger_HLT_Mu17_TrkIsoVVL_TkMu8_TrkIsoVVL_DZ || trigger_HLT_Mu17_TrkIsoVVL_Mu8_TrkIsoVVL_DZ)',
                'SingleMuon': '(trigger_HLT_IsoMu24 || trigger_HLT_IsoTkMu24) && !(trigger_HLT_Mu17_TrkIsoVVL_Mu8_TrkIsoVVL || trigger_HLT_Mu17_TrkIsoVVL_TkMu8_TrkIsoVVL || trigger_emu_flag)',
                'SingleMuon_2016_H': '(trigger_HLT_IsoMu24 || trigger_HLT_IsoTkMu24) && !(trigger_HLT_Mu17_TrkIsoVVL_TkMu8_TrkIsoVVL_DZ || trigger_HLT_Mu17_TrkIsoVVL_Mu8_TrkIsoVVL_DZ || trigger_emu_flag)',
                'SingleElectron': 'trigger_HLT_Ele32_WPTight_Gsf && !(trigger_HLT_Ele23_Ele12_CaloIdL_TrackIdL_IsoVL_DZ || trigger_emu_flag)',
                'DoubleLepton': '(trigger_HLT_Mu23_TrkIsoVVL_Ele12_CaloIdL_TrackIdL_IsoVL || trigger_HLT_Mu8_TrkIsoVVL_Ele23_CaloIdL_TrackIdL_IsoVL)',
                'DoubleLepton_2016_H': '(trigger_HLT_Mu8_TrkIsoVVL_Ele23_CaloIdL_TrackIdL_IsoVL_DZ || trigger_HLT_Mu23_TrkIsoVVL_Ele12_CaloIdL_TrackIdL_IsoVL_DZ)',
                'DoubleEG': 'trigger_HLT_Ele23_Ele12_CaloIdL_TrackIdL_IsoVL_DZ',
                'ee': 'trigger_ee_flag',
                'emu': 'trigger_emu_flag',
                'mumu': 'trigger_mumu_flag',
                'single_muon': 'trigger_HLT_IsoMu24'
            },
        '2017': 
            {
                'DoubleMuon': '(trigger_HLT_Mu17_TrkIsoVVL_Mu8_TrkIsoVVL_DZ_Mass8 || trigger_HLT_Mu17_TrkIsoVVL_Mu8_TrkIsoVVL_DZ_Mass3p8)',
                'SingleMuon': 'trigger_HLT_IsoMu27 && !(trigger_HLT_Mu17_TrkIsoVVL_Mu8_TrkIsoVVL_DZ_Mass8 || trigger_HLT_Mu17_TrkIsoVVL_Mu8_TrkIsoVVL_DZ_Mass3p8 || trigger_emu_flag)',
                'SingleElectron': 'trigger_HLT_Ele32_WPTight_Gsf && !(trigger_HLT_Ele23_Ele12_CaloIdL_TrackIdL_IsoVL || trigger_HLT_DoubleEle33_CaloIdL_MW || trigger_HLT_DoubleEle25_CaloIdL_MW || trigger_emu_flag)',
                'DoubleLepton': '(trigger_HLT_Mu23_TrkIsoVVL_Ele12_CaloIdL_TrackIdL_IsoVL || trigger_HLT_Mu8_TrkIsoVVL_Ele23_CaloIdL_TrackIdL_IsoVL_DZ)',
                'DoubleEG': '(trigger_HLT_Ele23_Ele12_CaloIdL_TrackIdL_IsoVL || trigger_HLT_DoubleEle33_CaloIdL_MW || trigger_HLT_DoubleEle25_CaloIdL_MW)', 
                'ee': 'trigger_ee_flag',
                'emu': 'trigger_emu_flag',
                'mumu': 'trigger_mumu_flag',
                'single_muon': 'trigger_HLT_IsoMu27'
            },
        '2018': 
            {
                'DoubleMuon': 'trigger_HLT_Mu17_TrkIsoVVL_Mu8_TrkIsoVVL_DZ_Mass3p8',
                'SingleMuon': 'trigger_HLT_IsoMu24 && !(trigger_HLT_Mu17_TrkIsoVVL_Mu8_TrkIsoVVL_DZ_Mass3p8 || trigger_emu_flag)',
                'DoubleLepton': '(trigger_HLT_Mu23_TrkIsoVVL_Ele12_CaloIdL_TrackIdL_IsoVL || trigger_HLT_Mu8_TrkIsoVVL_Ele23_CaloIdL_TrackIdL_IsoVL_DZ)',
                'DoubleEG': 'trigger_ee_flag && !trigger_emu_flag',
                'ee': 'trigger_ee_flag',
                'emu': 'trigger_emu_flag',
                'mumu': 'trigger_mumu_flag',
                'single_muon': 'trigger_HLT_IsoMu24'
            }, 
        '2022': 
            {
                'DoubleMuon': 'trigger_HLT_Mu17_TrkIsoVVL_Mu8_TrkIsoVVL_DZ_Mass3p8',
                'Muon': 'trigger_mumu_flag && !trigger_emu_flag',
                'DoubleLepton': 'trigger_emu_flag',
                'DoubleEG': 'trigger_ee_flag && !trigger_emu_flag',
                'ee': 'trigger_ee_flag',
                'emu': 'trigger_emu_flag',
                'mumu': 'trigger_mumu_flag',
                'single_muon': 'trigger_HLT_IsoMu24'
            }, 
        '2022EE': 
            {
                'Muon': 'trigger_mumu_flag && !trigger_emu_flag',
                'DoubleLepton': 'trigger_emu_flag',
                'DoubleEG': 'trigger_ee_flag && !trigger_emu_flag',
                'ee': 'trigger_ee_flag',
                'emu': 'trigger_emu_flag',
                'mumu': 'trigger_mumu_flag',
                'single_muon': 'trigger_HLT_IsoMu24'
            }, 
        }

    # --- trigger selection
    if is_data:
        process_tier = process.split("_", 1)[0]
        if '2016_H' in process and ('Muon' in process or 'Lepton' in process):
            process_tier = process
    else:
        process_tier = lepton_selection
    if lepton_selection == 'single_muon':
        process_tier = 'single_muon'
    filter_conditions = [f"{TRIGGER_SEL[year][process_tier]}"]

    # if lepton_selection == 'mumu' and '2016_H' in process:
    #     filter_conditions[0] = '(trigger_HLT_Mu17_TrkIsoVVL_TkMu8_TrkIsoVVL_DZ || trigger_HLT_Mu17_TrkIsoVVL_Mu8_TrkIsoVVL_DZ)'
    # if lepton_selection == 'emu' and '2016_H' in process:
    #     filter_conditions[0] = '(trigger_HLT_Mu8_TrkIsoVVL_Ele23_CaloIdL_TrackIdL_IsoVL_DZ || trigger_HLT_Mu23_TrkIsoVVL_Ele12_CaloIdL_TrackIdL_IsoVL_DZ)'

    # --- lepton selection
    if lepton_selection == 'ee':
        filter_conditions += [
            f"n{LEPTON_ID_ELE}_{ELECTRON_ID_TYPE}_Electrons == {n_leptons}"
            f" && {LEPTON_ID_ELE}_{ELECTRON_ID_TYPE}_Electrons_pt_leading > 25 && {LEPTON_ID_ELE}_{ELECTRON_ID_TYPE}_Electrons_pt_subleading > 15"
            f" && dilepton_invariant_{lepton_selection}_mass > 20",
            f"({LEPTON_ID_ELE}_{ELECTRON_ID_TYPE}_Electrons_charge_leading * {LEPTON_ID_ELE}_{ELECTRON_ID_TYPE}_Electrons_charge_subleading) < 1"
        ]
        if on_Z_peak == 'on':
            filter_conditions.append(
                f"(dilepton_invariant_{lepton_selection}_mass >= {Z_PEAK_LOW_EDGE} && dilepton_invariant_{lepton_selection}_mass <= {Z_PEAK_HIGH_EDGE})"
            )
        elif on_Z_peak == 'off':
            filter_conditions.append(
                f"(dilepton_invariant_{lepton_selection}_mass < {Z_PEAK_LOW_EDGE} || dilepton_invariant_{lepton_selection}_mass > {Z_PEAK_HIGH_EDGE})"
            )
    
    elif lepton_selection == 'mumu':
        filter_conditions += [
            f"ntightRelIso_{LEPTON_ID_MUON}ID_Muons == {n_leptons}"
            f" && tightRelIso_{LEPTON_ID_MUON}ID_Muons_pt_leading > 25 && tightRelIso_{LEPTON_ID_MUON}ID_Muons_pt_subleading > 15"
            f" && dilepton_invariant_{lepton_selection}_mass > 20",
            f" (tightRelIso_{LEPTON_ID_MUON}ID_Muons_charge_leading * tightRelIso_{LEPTON_ID_MUON}ID_Muons_charge_subleading) < 1"
        ]
        if on_Z_peak == 'on':
            filter_conditions.append(
                f"(dilepton_invariant_{lepton_selection}_mass >= {Z_PEAK_LOW_EDGE} && dilepton_invariant_{lepton_selection}_mass <= {Z_PEAK_HIGH_EDGE})"
            )
        elif on_Z_peak == 'off':
            filter_conditions.append(
                f"(dilepton_invariant_{lepton_selection}_mass < {Z_PEAK_LOW_EDGE} || dilepton_invariant_{lepton_selection}_mass > {Z_PEAK_HIGH_EDGE})"
            )

    elif lepton_selection == 'emu':
        filter_conditions += [
            f"n{LEPTON_ID_ELE}_{ELECTRON_ID_TYPE}_Electrons == 1 && ntightRelIso_{LEPTON_ID_MUON}ID_Muons == 1"
            f" && (({LEPTON_ID_ELE}_{ELECTRON_ID_TYPE}_Electrons_pt_leading > 25 && tightRelIso_{LEPTON_ID_MUON}ID_Muons_pt_leading > 15) "
            f"|| ({LEPTON_ID_ELE}_{ELECTRON_ID_TYPE}_Electrons_pt_leading >  15 && tightRelIso_{LEPTON_ID_MUON}ID_Muons_pt_leading > 25))",
            f" ({LEPTON_ID_ELE}_{ELECTRON_ID_TYPE}_Electrons_charge_leading * tightRelIso_{LEPTON_ID_MUON}ID_Muons_charge_leading) < 1"
        ]

    elif lepton_selection == 'single_muon':
        filter_conditions += [
            f"ntightRelIso_{LEPTON_ID_MUON}ID_Muons == 1",
            f"tightRelIso_{LEPTON_ID_MUON}ID_Muons_pt_leading > 50",
            f"n{LEPTON_ID_ELE}_{ELECTRON_ID_TYPE}_Electrons == 0"
        ]
        # --- leptonic W
        filter_conditions.append(
            f"(MET_energy + tightRelIso_{LEPTON_ID_MUON}ID_Muons_pt_leading > 250)"
        )
        # --- hadronic/leptonic top
        filter_conditions.append(
            f"(is_BJets_{ak4_systematic}_{b_wp}_outside_{boosted_jets}_and_tightRelIso_{LEPTON_ID_MUON}ID_Muons_in_same_hemisphere) && "
            f"(is_selectedHOTVRJets_{boosted_systematic}_and_tightRelIso_{LEPTON_ID_MUON}ID_Muons_in_same_hemisphere == 0) "
        )

    # --- jet selection + MET filter (and HEM veto)
    if n_ak4_outside is not None:
        additional_str = '_valid'
        if n_b_outside is None:
            additional_str = f'_outside_{boosted_jets}'

        if n_ak4_outside >= 2:
            filter_conditions.append(
                f"nselectedJets_{ak4_systematic}{additional_str}>={n_ak4_outside}"  #
            )
        else:
            filter_conditions.append(
                f"nselectedJets_{ak4_systematic}{additional_str}=={n_ak4_outside}" #
            )

    if n_b_outside is not None:
        if n_b_outside >= 2:
            filter_conditions.append(
                f"nBJets_{ak4_systematic}_{b_wp}_outside_{boosted_jets}>={n_b_outside}"
            )
        else:
            if lepton_selection =='single_muon':
                filter_conditions.append(
                    f"nBJets_{ak4_systematic}_{b_wp}_outside_{boosted_jets}>={n_b_outside}"
                )
            else:
                filter_conditions.append(
                    f"nBJets_{ak4_systematic}_{b_wp}_outside_{boosted_jets}=={n_b_outside}"
                )

    # --- noisy filter
    # filter_conditions.append(f'MET_filter')
    filter_conditions[-1] += ' && MET_filter '
    # ---

    # --- hem veto
    if year == '2018':
        # filter_conditions.append(f'HEM_veto')
        filter_conditions[-1] += '&& HEM_veto '
    # if year == '2018':
    #     filter_conditions.append(f'HEM_veto_only_hotvr_no_ak4')
    #     filter_conditions[-1] += '&& HEM_veto_only_hotvr_no_ak4 '
    # ---

    print(f'Conditions: {filter_conditions}')

    cumulative_string_filter = " && ".join(filter_conditions)
    print(f"\nSelection filter: {cumulative_string_filter}")

    root_df_filtered = root_df.Filter(cumulative_string_filter, f'event_selection_{lepton_selection}')

    if return_cutflow:
        # Determine weight use
        use_weights = not ('Double' in process or 'Muon' in process or 'Single' in process)
        
        cutflow_unweighted = []
        cutflow_weighted = []

        cumulative_condition = ""
        for i, condition in enumerate(filter_conditions):
            cumulative_condition = condition if cumulative_condition == "" else f"{cumulative_condition} && {condition}"
            label = f"Cut_{i+1}"
            current_df = root_df.Filter(cumulative_condition, label)

            n_unweighted = current_df.Count().GetValue()
            cutflow_unweighted.append((label, n_unweighted))

            if use_weights:
                if not current_df.HasColumn(f"{weight_column}"):
                    raise RuntimeError(f"Weight column {weight_column} not found in DataFrame. Did you define it?")
                n_weighted = current_df.Sum(f"{weight_column}").GetValue()
                cutflow_weighted.append((label, n_weighted))

        return root_df.Filter(cumulative_condition, f'event_selection_{lepton_selection}'), cutflow_unweighted, cutflow_weighted

    return root_df_filtered


def adding_event_selection(root_df_filtered, selection):
    root_df_filtered = root_df_filtered.Filter(
        selection, selection
    )
    return root_df_filtered


def adding_new_columns(
        root_df, 
        LEPTON_ID_ELE='medium', 
        ELECTRON_ID_TYPE='MVA', 
        LEPTON_ID_MUON='loose', 
        year='2018', 
        is_data=False, 
        BDT_SCORE_WPs=[], 
        ak4_systematic='nominal', 
        boosted_systematic='nominal', 
        process='DoubleMuon',
        nboosted_jets=2,
        ntagged_jets=2,
        is_bkg_estimation=False,
        is_jet_composition=False,
        is_cut_flow=False,
        on_Z_peak='on',
):

    #### new columns definitions
    # --- HEM veto
    if year == '2018':
        root_df = root_df.Define(
            "HEM_veto",
            f"HEM_veto("
            f'{LEPTON_ID_ELE}_{ELECTRON_ID_TYPE}_Electrons_pt, '
            f'{LEPTON_ID_ELE}_{ELECTRON_ID_TYPE}_Electrons_eta, '
            f'{LEPTON_ID_ELE}_{ELECTRON_ID_TYPE}_Electrons_phi, '
            f'selectedJets_noJEC_pt, '
            f'selectedJets_noJEC_eta, '
            f'selectedJets_noJEC_phi, '
            f'run, {"true" if is_data else "false"}'
            ")"
        )

        # root_df = root_df.Define(
        #     "HEM_veto_only_hotvr_no_ak4",
        #     f"HEM_veto_only_hotvr_no_ak4("
        #     f'selectedHOTVRJets_noJEC_pt, '
        #     f'selectedHOTVRJets_noJEC_eta, '
        #     f'selectedHOTVRJets_noJEC_phi, '
        #     f'selectedJets_noJEC_pt, '
        #     f'selectedJets_noJEC_eta, '
        #     f'selectedJets_noJEC_phi, '
        #     f'run, {"true" if is_data else "false"}'
        #     ")"
        # )
    # ---


    # ---invariant di-lepton mass
    root_df = root_df.Define(
        "dilepton_invariant_ee_mass",
        f'invariant_mass({LEPTON_ID_ELE}_{ELECTRON_ID_TYPE}_Electrons_pt, '
        f'{LEPTON_ID_ELE}_{ELECTRON_ID_TYPE}_Electrons_eta, '
        f'{LEPTON_ID_ELE}_{ELECTRON_ID_TYPE}_Electrons_phi, '
        f'{LEPTON_ID_ELE}_{ELECTRON_ID_TYPE}_Electrons_mass)'
    )
    root_df = root_df.Define(
        "dilepton_invariant_mumu_mass",
        f"invariant_mass(tightRelIso_{LEPTON_ID_MUON}ID_Muons_pt, "
        f"tightRelIso_{LEPTON_ID_MUON}ID_Muons_eta, "
        f"tightRelIso_{LEPTON_ID_MUON}ID_Muons_phi, "
        f"tightRelIso_{LEPTON_ID_MUON}ID_Muons_mass)"
    )
    root_df = root_df.Define(
        "dilepton_invariant_emu_mass",
        f"invariant_mass_emu({LEPTON_ID_ELE}_{ELECTRON_ID_TYPE}_Electrons_pt, "
        f"{LEPTON_ID_ELE}_{ELECTRON_ID_TYPE}_Electrons_eta, "
        f"{LEPTON_ID_ELE}_{ELECTRON_ID_TYPE}_Electrons_phi, "
        f"{LEPTON_ID_ELE}_{ELECTRON_ID_TYPE}_Electrons_mass,"
        f"tightRelIso_{LEPTON_ID_MUON}ID_Muons_pt, "
        f"tightRelIso_{LEPTON_ID_MUON}ID_Muons_eta, "
        f"tightRelIso_{LEPTON_ID_MUON}ID_Muons_phi, "
        f"tightRelIso_{LEPTON_ID_MUON}ID_Muons_mass)"
    )
    # ---

    # ---lepton objects
    lepton_variables = [
        'phi', 'eta', 'charge', 'pt', #'dxy', 'dz', 'pfRelIso04'
    ]
    # electron variables
    for var in lepton_variables + ['dxy', 'dz']:
        for ijet, jet in enumerate(['leading', 'subleading']): 
            root_df = root_df.Define(
                f"{LEPTON_ID_ELE}_{ELECTRON_ID_TYPE}_Electrons_{var}_{jet}",
                f"{LEPTON_ID_ELE}_{ELECTRON_ID_TYPE}_Electrons_{var}.size() > {ijet} ? {LEPTON_ID_ELE}_{ELECTRON_ID_TYPE}_Electrons_{var}[{ijet}] : -99"
            ) # always checking the size of the array; if empty, segmentation violation error arises
    # muon variables
    for var in lepton_variables + ['pfRelIso04_all']:
        for ijet, jet in enumerate(['leading', 'subleading']):
            root_df = root_df.Define(
                f"tightRelIso_{LEPTON_ID_MUON}ID_Muons_{var}_{jet}",
                f"tightRelIso_{LEPTON_ID_MUON}ID_Muons_{var}.size() > {ijet} ? tightRelIso_{LEPTON_ID_MUON}ID_Muons_{var}[{ijet}] : -99"
            ) # always checking the size of the array; if empty, segmentation violation error arises
    root_df = root_df.Define(
        'met_and_muon_pt',
        f'tightRelIso_{LEPTON_ID_MUON}ID_Muons_pt_leading + MET_energy'
    )
    # ---

    jet_variables = [
        'pt', 
        'phi', 
        'eta', 
        'mass', 
        'btagDeepFlavB', 
        'min_deltaRVShotvrJet', 
        'invariant_mass_leading_subleading', 
        'jetId',
        'puId'
    ]
    # --- ak4
    for var in jet_variables:
        if var == 'ht': continue
        if var == 'invariant_mass_leading_subleading':
            root_df = root_df.Define(
                f"selectedJets_{ak4_systematic}_{var}_{jet}",
                f"invariant_mass(selectedJets_{ak4_systematic}_pt, "
                f"selectedJets_{ak4_systematic}_eta, "
                f"selectedJets_{ak4_systematic}_phi, "
                f"selectedJets_{ak4_systematic}_mass)"
            )
            continue
        for ijet, jet in enumerate(['leading', 'subleading']): 
            root_df = root_df.Define(
                f"selectedJets_{ak4_systematic}_{var}_{jet}",
                f"selectedJets_{ak4_systematic}_{var}.size() > {ijet} ? selectedJets_{ak4_systematic}_{var}[{ijet}] : -99"
            )
    # ---b-jets
    for WP in B_TAGGING_WP[str(year)].keys():
        root_df = root_df.Define(
            f"BJets_{ak4_systematic}_{WP}", 
            f"selectedJets_{ak4_systematic}_btagDeepFlavB>{B_TAGGING_WP[str(year)][WP]}"
        )
        root_df = root_df.Define(
            f"nBJets_{ak4_systematic}_{WP}",
            f"Sum(BJets_{ak4_systematic}_{WP})"
            )
        
        root_df = root_df.Define(
            f"antiBJets_{ak4_systematic}_{WP}",  
            f"selectedJets_{ak4_systematic}_btagDeepFlavB<={B_TAGGING_WP[str(year)][WP]}"
        )
        root_df = root_df.Define(
            f"nantiBJets_{ak4_systematic}_{WP}",
            f"Sum(antiBJets_{ak4_systematic}_{WP})"
            )
        
        for var in jet_variables:
            if var == 'btagDeepFlavB': continue # or var == 'min_deltaRVShotvrJet': continue
            if var == 'invariant_mass_leading_subleading':
                root_df = root_df.Define(
                        f"BJets_{ak4_systematic}_{WP}_{var}",
                        f"invariant_mass(BJets_{ak4_systematic}_{WP}_pt, "
                        f"BJets_{ak4_systematic}_{WP}_eta, "
                        f"BJets_{ak4_systematic}_{WP}_phi, "
                        f"BJets_{ak4_systematic}_{WP}_mass)"
                )
                continue
            root_df = root_df.Define(
                f"BJets_{ak4_systematic}_{WP}_{var}",
                f"selectedJets_{ak4_systematic}_{var}[BJets_{ak4_systematic}_{WP}]"
            )
            for ijet, jet in enumerate(['leading', 'subleading']): 
                root_df = root_df.Define(
                    f"BJets_{ak4_systematic}_{WP}_{var}_{jet}",
                    f"BJets_{ak4_systematic}_{WP}_{var}.size() > {ijet} ? BJets_{ak4_systematic}_{WP}_{var}[{ijet}] : -99"
                )

        root_df = root_df.Define(
            f"is_BJets_{ak4_systematic}_{WP}_and_tightRelIso_{LEPTON_ID_MUON}ID_Muons_in_same_hemisphere",
            f"(BJets_{ak4_systematic}_{WP}_phi.size() > 0 && tightRelIso_{LEPTON_ID_MUON}ID_Muons_phi.size() > 0) ? "
            f"(abs(BJets_{ak4_systematic}_{WP}_phi[0] - tightRelIso_{LEPTON_ID_MUON}ID_Muons_phi[0]) < 1.2) : 0"
        )

        root_df = root_df.Define(
            f"delta_phi_BJets_{ak4_systematic}_{WP}_and_tightRelIso_{LEPTON_ID_MUON}ID_Muons",
            f"(BJets_{ak4_systematic}_{WP}_phi.size() > 0 && tightRelIso_{LEPTON_ID_MUON}ID_Muons_phi.size() > 0) ? "
            f"abs(BJets_{ak4_systematic}_{WP}_phi[0] - tightRelIso_{LEPTON_ID_MUON}ID_Muons_phi[0]) : -999"
        )
    # ---

    # --- cleaned ak4 + jetID + pUId (jet < 50)
    root_df = root_df.Define(
        f"selectedJets_{ak4_systematic}_in_hotvr",
        f'is_inside_hotvr('
        f'selectedJets_{ak4_systematic}_eta, '
        f'selectedJets_{ak4_systematic}_phi, '
        f'selectedHOTVRJets_{boosted_systematic}_pt, '
        f'selectedHOTVRJets_{boosted_systematic}_eta, '
        f'selectedHOTVRJets_{boosted_systematic}_phi, '
        f'{nboosted_jets}'
        f')'
    )
    if year == '2022' or year == '2022EE': # no need for PUjetID for Run 3 (PUPPI jets)
        root_df = root_df.Define(
            f"selectedJets_{ak4_systematic}_valid",
            f"(ROOT::VecOps::RVec<int>(selectedJets_{ak4_systematic}_jetId) & 0b10) != 0"
        )
    else:
        root_df = root_df.Define(
            f"selectedJets_{ak4_systematic}_valid",
            f"""
                ((ROOT::VecOps::RVec<int>(selectedJets_{ak4_systematic}_jetId) & 0b10) != 0) &&
                (
                    (selectedJets_{ak4_systematic}_pt < 50 && selectedJets_{ak4_systematic}_puId != 0) ||
                    (selectedJets_{ak4_systematic}_pt >= 50)
                )
            """
        )

    root_df = root_df.Define(
        f"nselectedJets_{ak4_systematic}_valid",
        f"Sum(selectedJets_{ak4_systematic}_valid)"
    )
    # for boosted_jets, boosted_jets_label in zip(['ak8', 'hotvr'], [f'selectedFatJets_{boosted_systematic}', f'selectedHOTVRJets_{boosted_systematic}']):
    for boosted_jets, boosted_jets_label in zip(['hotvr'], [f'selectedHOTVRJets_{boosted_systematic}']):
        for jet_type, jet_type_label in zip(['==1', '==0'], ['inside', 'outside']):
            # ---ak4
            root_df = root_df.Define(
                f"selectedJets_{ak4_systematic}_{jet_type_label}_{boosted_jets}",
                f"""
                    (ROOT::VecOps::RVec<int>(selectedJets_{ak4_systematic}_in_{boosted_jets}{jet_type})) 
                    &&
                    (selectedJets_{ak4_systematic}_valid)
                """
            )

            root_df = root_df.Define(
                f"nselectedJets_{ak4_systematic}_{jet_type_label}_{boosted_jets}", 
                f"Sum(selectedJets_{ak4_systematic}_{jet_type_label}_{boosted_jets})"
            )

            for var in jet_variables:
                if var == 'ht': continue
                if var == 'invariant_mass_leading_subleading':
                    root_df = root_df.Define(
                        f"selectedJets_{ak4_systematic}_{jet_type_label}_{boosted_jets}_{var}", 
                        f"invariant_mass(selectedJets_{ak4_systematic}_{jet_type_label}_{boosted_jets}_pt, selectedJets_{ak4_systematic}_{jet_type_label}_{boosted_jets}_eta, selectedJets_{ak4_systematic}_{jet_type_label}_{boosted_jets}_phi, selectedJets_{ak4_systematic}_{jet_type_label}_{boosted_jets}_mass)"
                    )
                    continue
                root_df = root_df.Define(
                    f"selectedJets_{ak4_systematic}_{jet_type_label}_{boosted_jets}_{var}", 
                    f"selectedJets_{ak4_systematic}_{var}[selectedJets_{ak4_systematic}_{jet_type_label}_{boosted_jets}]"
                )

                for ijet, jet in enumerate(['leading', 'subleading']): 
                    root_df = root_df.Define(
                        f"selectedJets_{ak4_systematic}_{jet_type_label}_{boosted_jets}_{var}_{jet}",
                        f"selectedJets_{ak4_systematic}_{jet_type_label}_{boosted_jets}_{var}.size() > {ijet} ? selectedJets_{ak4_systematic}_{jet_type_label}_{boosted_jets}_{var}[{ijet}] : -99"
                    )
            # ---

            # ---ak4 b-tagged
            for WP in B_TAGGING_WP[str(year)].keys():
                root_df = root_df.Define(
                    f"BJets_{ak4_systematic}_{WP}_{jet_type_label}_{boosted_jets}",
                    f"""
                        (ROOT::VecOps::RVec<int>(selectedJets_{ak4_systematic}_in_{boosted_jets}{jet_type}))
                        &&
                        (selectedJets_{ak4_systematic}_btagDeepFlavB>{B_TAGGING_WP[str(year)][WP]})
                        &&
                        (selectedJets_{ak4_systematic}_valid)
                    """
                ) 
                root_df = root_df.Define(
                    f"nBJets_{ak4_systematic}_{WP}_{jet_type_label}_{boosted_jets}", 
                    f"Sum(BJets_{ak4_systematic}_{WP}_{jet_type_label}_{boosted_jets})"
                )

                root_df = root_df.Define("antiBJets_{}_{}_{}_{}".format(ak4_systematic, WP, jet_type_label, boosted_jets), 
                                        "selectedJets_{}_btagDeepFlavB<={} && selectedJets_{}_is_inside_{}{}".format(
                                            ak4_systematic, B_TAGGING_WP[str(year)][WP], ak4_systematic, boosted_jets, jet_type))
                root_df = root_df.Define("nantiBJets_{}_{}_{}_{}".format(ak4_systematic, WP, jet_type_label, boosted_jets), 
                                        "Sum(antiBJets_{}_{}_{}_{})".format(ak4_systematic, WP, jet_type_label, boosted_jets))

                for var in jet_variables:
                    if var == 'ht': continue
                    if var == 'invariant_mass_leading_subleading':
                        root_df = root_df.Define(
                            f"BJets_{ak4_systematic}_{WP}_{jet_type_label}_{boosted_jets}_{var}", 
                            f"invariant_mass(BJets_{ak4_systematic}_{WP}_{jet_type_label}_{boosted_jets}_pt, BJets_{ak4_systematic}_{WP}_{jet_type_label}_{boosted_jets}_eta, BJets_{ak4_systematic}_{WP}_{jet_type_label}_{boosted_jets}_phi, BJets_{ak4_systematic}_{WP}_{jet_type_label}_{boosted_jets}_mass)"
                        )
                        continue

                    root_df = root_df.Define(
                        f"BJets_{ak4_systematic}_{WP}_{jet_type_label}_{boosted_jets}_{var}", 
                        f"selectedJets_{ak4_systematic}_{var}[BJets_{ak4_systematic}_{WP}_{jet_type_label}_{boosted_jets}]"
                    )

                    for ijet, jet in enumerate(['leading', 'subleading']): 
                        root_df = root_df.Define(
                            "BJets_{}_{}_{}_{}_{}_{}".format(ak4_systematic, WP, jet_type_label, boosted_jets, var, jet),
                            "BJets_{}_{}_{}_{}_{}.size() > {} ? BJets_{}_{}_{}_{}_{}[{}] : -99".format(
                                ak4_systematic, WP, jet_type_label, boosted_jets, var, ijet, ak4_systematic, WP, jet_type_label, boosted_jets, var, ijet
                            )
                        )
                        # print("selectedBJets_nominal_{}_outside_{}_{}_{}".format(WP, boosted_jets, var, jet))

                root_df = root_df.Define(
                    f"is_BJets_{ak4_systematic}_{WP}_{jet_type_label}_{boosted_jets}_and_tightRelIso_{LEPTON_ID_MUON}ID_Muons_in_same_hemisphere",
                    f"(BJets_{ak4_systematic}_{WP}_{jet_type_label}_{boosted_jets}_phi.size() > 0 && tightRelIso_{LEPTON_ID_MUON}ID_Muons_phi.size()) > 0 ? "
                    f"abs(BJets_{ak4_systematic}_{WP}_{jet_type_label}_{boosted_jets}_phi[0] - tightRelIso_{LEPTON_ID_MUON}ID_Muons_phi[0]) < 1.2 : 0"
                )
                root_df = root_df.Define(
                    f"delta_phi_BJets_{ak4_systematic}_{WP}_{jet_type_label}_{boosted_jets}_and_tightRelIso_{LEPTON_ID_MUON}ID_Muons",
                    f"(BJets_{ak4_systematic}_{WP}_{jet_type_label}_{boosted_jets}_phi.size() > 0 && tightRelIso_{LEPTON_ID_MUON}ID_Muons_phi.size()) > 0 ? "
                    f"abs(BJets_{ak4_systematic}_{WP}_{jet_type_label}_{boosted_jets}_phi[0] - tightRelIso_{LEPTON_ID_MUON}ID_Muons_phi[0]) : -999"
                )

        root_df = root_df.Define("ht_ak4_and_{}".format(boosted_jets), 
                                "HT({}_pt, selectedJets_{}_outside_{}_pt)".format(boosted_jets_label, ak4_systematic, boosted_jets))
        root_df = root_df.Define("ht_ak4_outside_{}".format(boosted_jets), 
                                "HT_only_ak4(selectedJets_{}_outside_{}_pt)".format(ak4_systematic, boosted_jets))
        # ---

    # ---boosted objects
    hotvr_variables = [
        'pt', 'mass',
        'tau3_over_tau2', 'scoreBDT', 'fractional_subjet_pt', 'phi', 'eta',
        'min_pairwise_subjets_mass', 'nsubjets', 'invariant_mass_leading_subleading', 
        'max_eta_subjets', 'corrFactor' #'index'
    ]
    root_df = root_df.Define("selectedHOTVRJets_{}_nsubjets_gt2".format(boosted_systematic),
                                "selectedHOTVRJets_{}_nsubjets >=2".format(boosted_systematic))
    root_df = root_df.Define("nselectedHOTVRJets_{}_nsubjets_gt2".format(boosted_systematic),
                                "Sum(selectedHOTVRJets_{}_nsubjets_gt2)".format(boosted_systematic))
    for var in hotvr_variables:
        if var == 'invariant_mass_leading_subleading': continue
        root_df = root_df.Define("selectedHOTVRJets_{}_nsubjets_gt2_{}".format(boosted_systematic, var), 
                                'selectedHOTVRJets_{}_{}[selectedHOTVRJets_{}_nsubjets_gt2]'.format(boosted_systematic, var, boosted_systematic))
    
    # --- cut-based tagging
    # root_df = root_df.Define(
    #     f"selectedHOTVRJets_{boosted_systematic}_is_top_tagged_cut_based",
    #     "hotvr_cut_based_top_tagged("
    #     "selectedHOTVRJets_{}_subJetIdx1, selectedHOTVRJets_{}_subJetIdx2, selectedHOTVRJets_{}_subJetIdx3, "
    #     "selectedHOTVRJets_{}_pt, selectedHOTVRJets_{}_eta, selectedHOTVRJets_{}_phi, selectedHOTVRJets_{}_mass, "
    #     "selectedHOTVRJets_{}_tau3, selectedHOTVRJets_{}_tau2, "
    #     "selectedHOTVRSubJets_{}_index, selectedHOTVRSubJets_{}_pt, selectedHOTVRSubJets_{}_eta, selectedHOTVRSubJets_{}_phi, selectedHOTVRSubJets_{}_mass)".format(
    #         boosted_systematic, boosted_systematic, boosted_systematic,
    #         boosted_systematic, boosted_systematic, boosted_systematic, boosted_systematic,
    #         boosted_systematic, boosted_systematic,
    #         boosted_systematic, boosted_systematic, boosted_systematic, boosted_systematic, boosted_systematic)
    #     )
    # root_df = root_df.Define("nselectedHOTVRJets_{}_is_top_tagged_cut_based".format(boosted_systematic), 
    #                          "Sum(selectedHOTVRJets_{}_is_top_tagged_cut_based)".format(boosted_systematic))
    # ---

    # --- AK8 top tagging
    # root_df = root_df.Define("selectedFatJets_{}_is_top_tagged".format(boosted_systematic),
    #                             "selectedFatJets_{}_particleNet_TvsQCD > 0.58".format(boosted_systematic))
    # root_df = root_df.Define("nselectedFatJets_{}_is_top_tagged".format(boosted_systematic), 
    #                          "Sum(selectedFatJets_{}_is_top_tagged)".format(boosted_systematic))
    # ---

    for var in hotvr_variables:
        if var == 'invariant_mass_leading_subleading':
            root_df = root_df.Define(
                "selectedHOTVRJets_{}_{}".format(boosted_systematic, var),
                "invariant_mass(selectedHOTVRJets_{}_pt, selectedHOTVRJets_{}_eta, selectedHOTVRJets_{}_phi, selectedHOTVRJets_{}_mass)".format(
                    boosted_systematic, boosted_systematic, boosted_systematic, boosted_systematic
                )
            )
        else:
            for ijet, jet in enumerate(['leading', 'subleading']): 
                root_df = root_df.Define(
                    "selectedHOTVRJets_{}_{}_{}".format(boosted_systematic, var, jet),
                    "selectedHOTVRJets_{}_{}.size() > {} ? selectedHOTVRJets_{}_{}[{}] : -99".format(
                        boosted_systematic, var, ijet, boosted_systematic, var, ijet)
                ) # always checking the size of the array; if empty, segmentation violation error arises
                root_df = root_df.Define(
                    "selectedHOTVRJets_{}_nsubjets_gt2_{}_{}".format(boosted_systematic, var, jet),
                    "selectedHOTVRJets_{}_nsubjets_gt2_{}.size() > {} ? selectedHOTVRJets_{}_nsubjets_gt2_{}[{}] : -99".format(
                        boosted_systematic, var, ijet, boosted_systematic, var, ijet)
                ) # always checking the size of the array; if empty, segmentation violation error arises

    root_df = root_df.Define(
        f"is_selectedHOTVRJets_{boosted_systematic}_and_tightRelIso_{LEPTON_ID_MUON}ID_Muons_in_same_hemisphere",
        f"(selectedHOTVRJets_{boosted_systematic}_phi.size() > 0 && tightRelIso_{LEPTON_ID_MUON}ID_Muons_phi.size()) > 0 ? "
        f"(abs(selectedHOTVRJets_{boosted_systematic}_phi[0] - tightRelIso_{LEPTON_ID_MUON}ID_Muons_phi[0]) < 2) : 0"
    )
    root_df = root_df.Define(
        f"delta_phi_selectedHOTVRJets_{boosted_systematic}_and_tightRelIso_{LEPTON_ID_MUON}ID_Muons",
        f"(selectedHOTVRJets_{boosted_systematic}_phi.size() > 0 && tightRelIso_{LEPTON_ID_MUON}ID_Muons_phi.size()) > 0 ? "
        f"abs(selectedHOTVRJets_{boosted_systematic}_phi[0] - tightRelIso_{LEPTON_ID_MUON}ID_Muons_phi[0]) : -999"
    )


    # --- BDT top tagging
    for wp in BDT_SCORE_WPs:
        if boosted_systematic == 'noJEC': continue
        wp_str = str(wp).replace('.', 'p')
        root_df = root_df.Define(
            f"selectedHOTVRJets_{boosted_systematic}_is_top_tagged_wp_{wp_str}",
            f"is_top_tagged(selectedHOTVRJets_{boosted_systematic}_scoreBDT, {wp})"
        )
        root_df = root_df.Define(
            f"nselectedHOTVRJets_{boosted_systematic}_is_top_tagged_wp_{wp_str}",
            f"Sum(selectedHOTVRJets_{boosted_systematic}_is_top_tagged_wp_{wp_str})"
        )

        for var in hotvr_variables:
            if var == 'invariant_mass_leading_subleading': continue

            root_df = root_df.Define(
                f"selectedHOTVRJets_{boosted_systematic}_is_top_tagged_wp_{wp_str}_{var}", 
                f"selectedHOTVRJets_{boosted_systematic}_{var}[selectedHOTVRJets_{boosted_systematic}_is_top_tagged_wp_{wp_str}]"
            )
            for ijet, jet in enumerate(['leading', 'subleading']):
                root_df = root_df.Define(
                    f"selectedHOTVRJets_{boosted_systematic}_is_top_tagged_wp_{wp_str}_{var}_{jet}", 
                    f"selectedHOTVRJets_{boosted_systematic}_is_top_tagged_wp_{wp_str}[{ijet}] ? "
                    f"selectedHOTVRJets_{boosted_systematic}_{var}[{ijet}] : -99"
                )
    
        # --- sf calculation
        if not is_data:
            for sys_type in ["nominal", "up", "down"]:
                root_df = root_df.Define(
                    f"bdt_sf_{sys_type}",
                    f"bdt_sf(selectedHOTVRJets_nominal_pt, "
                    f"selectedHOTVRJets_nominal_has_hadronicTop_topIsInside, "
                    # f"selectedHOTVRJets_nominal_has_genTopHadronic_inside, "
                    f'"{year}", '
                    f'"{sys_type}")'
                )
        # ---
    # --- 

    # --- PU ID SFs (Run2) (loose WP)
    if year != '2022' and year != '2022EE':
        histo_eff_name = f'h2_eff_mcUL{year}_L' if year != '2016preVFP' else f'h2_eff_mcUL2016APV_L'
        histo_sf_name = f'h2_eff_sfUL{year}_L' if year != '2016preVFP' else f'h2_eff_sfUL2016APV_L'
        root_df = root_df.Define(
            f"puID_L_sf_and_error",
            f'get_PUID_scale_factor::rescaling_weights('
            'selectedJets_nominal_puId,'
            'selectedJets_nominal_pt,'
            'selectedJets_nominal_eta,'
            f'"{histo_eff_name}",'
            f'"{histo_sf_name}")'
        )

        root_df = root_df.Define(f"puID_L_sf", "puID_L_sf_and_error.first")
        root_df = root_df.Define(f"puID_L_sf_error", "puID_L_sf_and_error.second")
        root_df = root_df.Define(f"puID_L_sf_up", "puID_L_sf + puID_L_sf_error")
        root_df = root_df.Define(f"puID_L_sf_down", "puID_L_sf - puID_L_sf_error")

    # ---

    # --- double trigger SFs (Run 3)
    if year == '2022' or year == '2022EE':
        for lepton_selection in ['ee', 'mumu', 'emu']:
            objects = ''
            if lepton_selection == 'ee':
                objects = f"{LEPTON_ID_ELE}_{ELECTRON_ID_TYPE}_Electrons_pt"
            elif lepton_selection == 'mumu':
                objects = f"tightRelIso_{LEPTON_ID_MUON}ID_Muons_pt"
            elif lepton_selection == 'emu':
                muon_pt = f"tightRelIso_{LEPTON_ID_MUON}ID_Muons_pt"
                ele_pt = f"{LEPTON_ID_ELE}_{ELECTRON_ID_TYPE}_Electrons_pt"
                # make an RVec<double> with [muon_pt[0], ele_pt[0]]
                objects = f"ROOT::VecOps::RVec<float>{{static_cast<float>({ele_pt}[0]), static_cast<float>({muon_pt}[0])}}" # the tagging efficiency is calculated as: x-axis -> electron pt; y-axis -> muon pt

            root_df = root_df.Define(
                f"trigger_sf_and_error_{lepton_selection}",
                f'get_trigger_scale_factor::rescaling_weights('
                f'{objects},'
                f'"{year}",'
                f'"SF_is_{lepton_selection.replace("mu", "m")}_pt1_pt2")'
            )

            root_df = root_df.Define(f"trigger_sf_{lepton_selection}", f"trigger_sf_and_error_{lepton_selection}.first")
            root_df = root_df.Define(f"trigger_sf_error_{lepton_selection}", f"trigger_sf_and_error_{lepton_selection}.second")
            root_df = root_df.Define(f"trigger_sf_up_{lepton_selection}", f"trigger_sf_{lepton_selection} + trigger_sf_error_{lepton_selection}")
            root_df = root_df.Define(f"trigger_sf_down_{lepton_selection}", f"trigger_sf_{lepton_selection} - trigger_sf_error_{lepton_selection}")

            # test = root_df.AsNumpy([f"trigger_sf_error_{lepton_selection}"])
            # print(test)
    # ---

    # additional variables
    for jet in [
        "selectedHOTVRJets_{}".format(boosted_systematic),
        "selectedHOTVRJets_{}_is_top_tagged_wp_{}".format(boosted_systematic, str(wp).replace('.', 'p'))
    ]:
        root_df = root_df.Define("{}_leading_plus_subleading_pt".format(jet), 
                             "{}_pt_leading + {}_pt_subleading".format(jet, jet))
        root_df = root_df.Define("{}_deltaR".format(jet),
                             "deltaR_jets({}_eta, {}_phi)".format(jet, jet))

    if is_data == False and is_jet_composition:
        #adhoc flag definition for JES/JER variation, not present in NANO
        if 'JER' in boosted_systematic or 'JES' in boosted_systematic:
            for flag in JET_COMPOSITION_FLAGS:
                root_df = root_df.Define(
                    f"selectedHOTVRJets_{boosted_systematic}_{flag}", 
                    f"jes_jer_flags_from_nominal_hotvr("
                    f"selectedHOTVRJets_nominal_eta, selectedHOTVRJets_nominal_phi,"
                    f"selectedHOTVRJets_nominal_pt, selectedHOTVRJets_nominal_{flag},"
                    f"selectedHOTVRJets_{boosted_systematic}_eta, selectedHOTVRJets_{boosted_systematic}_phi,"
                    f"selectedHOTVRJets_{boosted_systematic}_pt)"
                )
        # sys.exit()
        
        # merging flags togeter into classes
        for composition_flag, jet_composition in JET_COMPOSITION_FLAGS_MERGED.items():
            root_df = root_df.Define(f"selectedHOTVRJets_{boosted_systematic}_flags_{composition_flag}", jet_composition)
            root_df = root_df.Define(
                f"selectedHOTVRJets_{boosted_systematic}_{composition_flag}",
                """
                ROOT::VecOps::RVec<int> int_flags;
                auto flags = selectedHOTVRJets_{boosted_systematic}_flags_{composition_flag};
                for (size_t i = 0; i < flags.size(); ++i) {{
                    int_flags.push_back(flags[i] ? 1 : 0);
                }}
                return int_flags;
                """.format(composition_flag=composition_flag, boosted_systematic=boosted_systematic)
            )
            root_df = root_df.Define(
                f"nselectedHOTVRJets_{boosted_systematic}_{composition_flag}",
                f"ROOT::VecOps::Sum(selectedHOTVRJets_{boosted_systematic}_{composition_flag})"
            )
            # ---
        # sys.exit()
            # ---
        for ijet, jet in enumerate(['leading', 'subleading']):
            for composition_flag, jet_composition in JET_COMPOSITION_FLAGS_MERGED.items():
                if '||' in jet_composition:
                    jet_composition = jet_composition.replace('||', f'[{ijet}]||')
                    jet_composition += f'[{ijet}]'
                else:
                    jet_composition += f'[{ijet}]'

                for var in hotvr_variables:
                    if var == 'invariant_mass_leading_subleading': 
                        continue
                    # --- uncovered jet type
                    if composition_flag == 'non_covered':
                        root_df = root_df.Define(
                            f"selectedHOTVRJets_{boosted_systematic}_{composition_flag}_{var}_{jet}", 
                            f"variable_per_jet_per_jet_composition_non_covered(selectedHOTVRJets_{boosted_systematic}_{var}, {jet_composition}, {ijet})"
                        )
                    # ---
                    else:
                        root_df = root_df.Define(
                            f"selectedHOTVRJets_{boosted_systematic}_{composition_flag}_{var}_{jet}", 
                            f"variable_per_jet_per_jet_composition(selectedHOTVRJets_{boosted_systematic}_{var}, {jet_composition}, {ijet})"
                        )
    # ---

    if is_bkg_estimation:
        root_df = root_df.Define(
            "tagging_efficiency_weight_and_error",
            f'get_scale_factor::rescaling_weights(\
                selectedHOTVRJets_{boosted_systematic}_pt, selectedHOTVRJets_{boosted_systematic}_mass, selectedHOTVRJets_{boosted_systematic}_scoreBDT, \
                {ntagged_jets}, "{year}", {"true" if is_data else "false"}\
            )'
        )
        root_df = root_df.Define("tagging_efficiency_weight", "tagging_efficiency_weight_and_error.first")
        root_df = root_df.Define("tagging_efficiency_weight_error", "tagging_efficiency_weight_and_error.second")
        root_df = root_df.Define("tagging_efficiency_weight_up", "tagging_efficiency_weight + tagging_efficiency_weight_error")
        root_df = root_df.Define("tagging_efficiency_weight_down", "tagging_efficiency_weight - tagging_efficiency_weight_error")

        # variables for background estimation
        for var in ['pt', 'mass']:
            if 'ttZJets' in process or 'ttWJets' in process or 'ttll' in process or 'ttlnu' in process:
                root_df = root_df.Define(
                    f"hotvr_combined_{var}", 
                    "ROOT::VecOps::RVec<float>{"
                    f"(selectedHOTVRJets_{boosted_systematic}_hadronic_t[0] == 1"
                    f"? selectedHOTVRJets_{boosted_systematic}_{var}_leading" 
                    ": static_cast<float>(0.0)),"
                    f"(selectedHOTVRJets_{boosted_systematic}_hadronic_t[1] == 1 "
                    f"? selectedHOTVRJets_{boosted_systematic}_{var}_subleading" 
                    ": static_cast<float>(0.0))}"
                )
                root_df = root_df.Define(
                    f"hotvr_combined_is_top_tagged_wp_{wp_str}_{var}", 
                    "ROOT::VecOps::RVec<float>{"
                    f"(selectedHOTVRJets_{boosted_systematic}_hadronic_t[0] == 1"
                    f"? selectedHOTVRJets_{boosted_systematic}_is_top_tagged_wp_{wp_str}_{var}_leading" 
                    ": static_cast<float>(0.0)),"
                    f"(selectedHOTVRJets_{boosted_systematic}_hadronic_t[1] == 1 "
                    f"? selectedHOTVRJets_{boosted_systematic}_is_top_tagged_wp_{wp_str}_{var}_subleading" 
                    ": static_cast<float>(0.0))}"
                )
            else:
                root_df = root_df.Define(
                    f"hotvr_combined_{var}",
                    f"""ROOT::VecOps::RVec<float>{{
                        selectedHOTVRJets_{boosted_systematic}_{var}_leading, 
                        selectedHOTVRJets_{boosted_systematic}_{var}_subleading
                    }}"""
                )
                root_df = root_df.Define(
                    f"hotvr_combined_is_top_tagged_wp_{wp_str}_{var}",
                    f"""ROOT::VecOps::RVec<float>{{
                        selectedHOTVRJets_{boosted_systematic}_is_top_tagged_wp_{wp_str}_{var}_leading,
                        selectedHOTVRJets_{boosted_systematic}_is_top_tagged_wp_{wp_str}_{var}_subleading
                    }}"""
                )
    # ---


    # --- genTop studies
    # if is_data == False:
    #     root_df = root_df.Define("genTop_hadronic_pt", "genTop_pt[genTop_has_hadronically_decay]")
    #     for boosted_jet in ['hotvr', 'ak8']:
    #         root_df = root_df.Define("genTop_is_inside_{}_top_tagged_pt".format(boosted_jet),
    #                                  "genTop_pt[genTop_is_inside_{}_top_tagged]".format(boosted_jet))
    #         root_df = root_df.Define("genTop_is_inside_{}_pt".format(boosted_jet),
    #                                  "genTop_pt[genTop_is_inside_{}]".format(boosted_jet))

    #         root_df = root_df.Define("genTop_hadronic_is_inside_{}_top_tagged_pt".format(boosted_jet),
    #                                  "genTop_pt[genTop_is_inside_{}_top_tagged & genTop_has_hadronically_decay]".format(boosted_jet))
    #         root_df = root_df.Define("genTop_hadronic_is_inside_{}_pt".format(boosted_jet),
    #                                  "genTop_pt[genTop_is_inside_{} & genTop_has_hadronically_decay]".format(boosted_jet))
            
    #         root_df = root_df.Define("genTop_hadronic_all_decays_is_inside_{}_top_tagged_pt".format(boosted_jet),
    #                                  "genTop_pt[genTop_is_inside_{}_top_tagged & genTop_has_hadronically_decay & genTop_all_decays_inside_{}]".format(boosted_jet, boosted_jet))
    #         root_df = root_df.Define("genTop_hadronic_all_decays_is_inside_{}_pt".format(boosted_jet),
    #                                  "genTop_pt[genTop_is_inside_{} & genTop_has_hadronically_decay & genTop_all_decays_inside_{}]".format(boosted_jet, boosted_jet))

    # #     root_df = root_df.Define("ngenTop_hadronically_decay", 
    #                              "Sum(genTop_has_hadronically_decay)")
    # # ---

    ####
    
    return root_df

def me_uncertainties_handler(root_df, tot_weights, nlhe):
    nlhescale_weights = root_df.Mean("nLHEScaleWeight").GetValue()
    if int(nlhescale_weights) == 9:
        return tot_weights, nlhe
    elif int(nlhescale_weights) == 8:
        tot_weights = tot_weights.replace("LHEScaleWeight[5]", "LHEScaleWeight[4]")
        tot_weights = tot_weights.replace("LHEScaleWeight[6]", "LHEScaleWeight[5]")
        tot_weights = tot_weights.replace("LHEScaleWeight[7]", "LHEScaleWeight[6]")
        tot_weights = tot_weights.replace("LHEScaleWeight[8]", "LHEScaleWeight[7]")

        if nlhe == 5: nlhe = 4
        if nlhe == 6: nlhe = 5
        if nlhe == 7: nlhe = 6
        if nlhe == 8: nlhe = 7
    else:
        print("Unexpected length of the norm for LHEScaleWeight")
    return tot_weights, nlhe
