import os, sys
import re
from argparse import ArgumentParser
from itertools import product

import yaml
from yaml.loader import SafeLoader

import numpy as np
from array import array
from collections import OrderedDict

import ROOT
ROOT.ROOT.EnableImplicitMT()

cpp_functions_header = "{}/cpp_functions_header.h".format(os.getcwd())
if not os.path.isfile(cpp_functions_header):
    print('No cpp header found!')
    sys.exit()
ROOT.gInterpreter.Declare('#include "{}"'.format(cpp_functions_header))

LUMINOSITY = {
    '2018': 59830, '2017': 41480,
    '2016preVFP': 19500, '2016': 16500
}

LEPTON_SELECTION = ['ee', 'emu', 'mumu']
ELECTRON_ID_TYPE = "MVA"
LEPTON_ID = "medium"

EVENT_SELECTION = "after_2OS_2ak4_2b_2ak8_tagged"

WP = [0.87] #, 0.95, 0.995]
# WP_CUTS_LABELS = {
#     (0.87, 0.87): 'lead_loose_sublead_loose', (0.87, 0.95): 'lead_loose_sublead_medium', (0.87, 0.995): 'lead_loose_sublead_tight',
#     (0.95, 0.87): 'lead_medium_sublead_loose', (0.95, 0.95): 'lead_medium_sublead_medium', (0.95, 0.995): 'lead_medium_sublead_tight',
#     (0.995, 0.87): 'lead_tight_sublead_loose', (0.995, 0.95): 'lead_tight_sublead_medium', (0.995, 0.995): 'lead_tight_sublead_tight',
# }
WP_CUTS_LABELS = {
    (0.87, 0.87): 'lead_loose_OR_sublead_loose',
    (0.87): 'sublead_loose'
}
B_TAGGING_WP = {
    '2018': '0.0490', '2017': '0.0532',
    '2016': '0.0480', '2016preVFP': '0.0614'
} 
PNET_CUT = {
    '2018': '0.58', '2017': '0.58',
    '2016': '0.5', '2016preVFP': '0.5'
}

WEIGHTS_DICT = {
    'ee': "event_weight * {} * {} * {} * {} ".format('trigger_weight_nominal', LEPTON_ID+"_"+ELECTRON_ID_TYPE+"_Electrons_weight_id_nominal", \
        LEPTON_ID+"_"+ELECTRON_ID_TYPE+"_Electrons_weight_recoPtAbove20_nominal", LEPTON_ID+"_"+ELECTRON_ID_TYPE+"_Electrons_weight_recoPtBelow20_nominal"), \
    'emu': "event_weight * {} * {} * {} * {} * {} * {}".format('trigger_weight_nominal', LEPTON_ID+"_"+ELECTRON_ID_TYPE+"_Electrons_weight_id_nominal", \
                LEPTON_ID+"_"+ELECTRON_ID_TYPE+"_Electrons_weight_recoPtAbove20_nominal", LEPTON_ID+"_"+ELECTRON_ID_TYPE+"_Electrons_weight_recoPtBelow20_nominal", \
                    "tightRelIso_"+LEPTON_ID+"ID_Muons_weight_id_nominal", "tightRelIso_"+LEPTON_ID+"ID_Muons_weight_iso_nominal"),\
    'mumu': "event_weight * {} * {} * {} ".format('trigger_weight_nominal',"tightRelIso_"+LEPTON_ID+"ID_Muons_weight_id_nominal", \
                            "tightRelIso_"+LEPTON_ID+"ID_Muons_weight_iso_nominal")}

SYSTEMATIC = 'nominal'

VARIABLES_BINNING = OrderedDict()
# {
#     'scoreBDT': list(np.arange(0., 1.005, .005)), 'pt': list(np.arange(0., 3020., 20)), 'mass': list(np.arange(0., 1510., 10)),  
#     'tau2_over_tau1': list(np.arange(0., 1.01, .01)), 
# } #'min_deltaR_vs_ak4': list(np.arange(0., 10.2, 0.2)), 'min_deltaR_vs_lepton': list(np.arange(0., 10.2, 0.2))
VARIABLES_BINNING['pt'] = list(np.arange(0., 3020., 20))
VARIABLES_BINNING['mass'] = list(np.arange(0., 1510., 10))
# VARIABLES_BINNING['tau2_over_tau1'] = list(np.arange(0., 1.01, .01))
# VARIABLES_BINNING['tau3_over_tau2'] = list(np.arange(0., 1.01, .01))
VARIABLES_BINNING['particleNet_TvsQCD'] = list(np.arange(0., 1.005, .005))
VARIABLES_BINNING['invariant_mass_leading_subleading'] = list(np.arange(0., 3020., 20))

SIGNAL_ONLY_HADRONIC_TOP = True

class Processor:
    def __init__(self, input_file, output_dir, year, is_data, is_sgn, weighting, sys):
        self.input_file = input_file
        self.output_dir = output_dir
        self.year = year
        self.is_data = is_data
        self.is_sgn = is_sgn
        self.weighting = weighting
        self.sys = sys  

        self.process_name = self._parsing_file()
        self.xsec = self._xsec()
        self.sum_gen_weights = self._sum_gen_weights()
        self.output_file = self._creation_output_file() 

        self.pattern = re.compile(r'selectedFatJets_nominal_(.*)')

    def _parsing_file(self):
        print("Processing file: {}".format(self.input_file))
        pattern = re.compile(r"merged\/(.*?)_MC")
        match = pattern.search(str(self.input_file))
        if match:
            print("Process name: {}".format(match.group(1)))
            return match.group(1)
        else:
            pattern = re.compile(r"merged\/(.*?)_ntuplizer")
            match = pattern.search(str(self.input_file))
            if match:
                print("Process name: {}".format(match.group(1)))
                return match.group(1)
            else:
                print("No process name found.")
                sys.exit()

    def _xsec(self):
        with open('{}/xsec.yaml'.format(os.getcwd())) as xsec_file:
            xsecFile = yaml.load(xsec_file, Loader=SafeLoader)
        if xsecFile[self.process_name]['isUsed']:
            return xsecFile[self.process_name]['xSec']
        else:
            print("Xsec for process {} not found in file".format(self.process_name))
            sys.exit()

    def _sum_gen_weights(self):
        if self.is_sgn:
            root_file = ROOT.TFile(str(self.input_file), 'READ')
            sumgenweight = root_file.Get("sumGenWeights")
            return sumgenweight.GetVal()
        else:
            with open("/nfs/dust/cms/user/gmilella/ttX_ntuplizer/bkg_{}_hotvr/merged/sum_gen_weights.yaml".format(self.year)) as sumGenWeights_file:
                sumGenWeightsFile = yaml.load(sumGenWeights_file, Loader=SafeLoader)
                return sumGenWeightsFile[self.process_name]

    def _creation_output_file(self):
        # bkg samples are divided in chunks
        # so the output files should reflect this division
        match = re.search(r'([^/]+)\.root$', str(self.input_file))
        if match: 
            process_filename = match.group(1)
        else:
            print("File name not extracted properly...")
            sys.exit()

        # check if dir exist
        if SIGNAL_ONLY_HADRONIC_TOP and self.is_sgn: 
            output_dir_path = os.path.join(self.output_dir, str(self.year), "ak8_variables_{}".format(EVENT_SELECTION), self.sys, "signal_only_hadronic_gen_top_from_resonance")
        else:
            output_dir_path = os.path.join(self.output_dir, str(self.year), "ak8_variables_{}".format(EVENT_SELECTION), self.sys)
        if not os.path.exists(output_dir_path):
            os.makedirs(output_dir_path)

        if SIGNAL_ONLY_HADRONIC_TOP and self.is_sgn: 
            output_path = os.path.join(output_dir_path, "{}_ak8_variables_{}_only_hadronic_gen_top_from_resonance.root".format(process_filename, self.sys)) #_only_hadronic_gen_top_from_resonance.root
        else:
            output_path = os.path.join(output_dir_path, "{}_ak8_variables_{}.root".format(process_filename, self.sys)) #_only_hadronic_gen_top_from_resonance.root

        file_out = ROOT.TFile(output_path, 'RECREATE')
        for lep_sel in ['emu', 'ee', 'mumu']:
            ROOT.gDirectory.mkdir(lep_sel)
            file_out.cd()
        print("Output file {}: ".format(file_out))
        return file_out

    def process(self):
        print("Process: {}, XSec: {} pb, Sum of gen weights: {}".format(self.process_name, self.xsec, self.sum_gen_weights))
        root_df = ROOT.RDataFrame("Friends", str(self.input_file))
        if not self.is_data:
            root_df = root_df.Define("event_weight", 
                                     "genweight * btagEventWeight_deepJet_shape_nominal * puWeight * {} / {} * {}".format(self.xsec, self.sum_gen_weights, LUMINOSITY[str(self.year)]))

        # adding new columns
        root_df = self._adding_new_columns(root_df)
        
        # event selection + histogram definitions
        for lepton_selection in LEPTON_SELECTION:
            print('\nLepton Selection: {}'.format(lepton_selection))
            
            root_df_filtered = self._event_selection(root_df, lepton_selection=lepton_selection)
            root_df_filtered = root_df_filtered.Define("weight", WEIGHTS_DICT[lepton_selection])

            self.output_file.cd(lepton_selection)

            root_df_filtered = self._adding_event_selection(root_df_filtered, "nselectedFatJets_nominal_top_tagged==2") # add filtering condition for plotting leading and subleading

            for var in VARIABLES_BINNING.keys():
                print('Variable --> {}'.format(var))
                if var == 'invariant_mass_leading_subleading':
                    output_histo = root_df_filtered.Histo1D(
                    ("{}_ak8_{}_{}_{}".format(self.process_name, var, EVENT_SELECTION, lepton_selection), '', 
                     len(VARIABLES_BINNING[var])-1, array('d', VARIABLES_BINNING[var])), 
                    'selectedFatJets_nominal_{}'.format(var), 'weight'
                    ) 
                    output_histo.Write()
                    print('Total events --> {}'.format(output_histo.Integral()))
                else: 
                    for ijet, jet in enumerate(['leading', 'subleading']): 
                        output_histo = root_df_filtered.Histo1D(
                            ("{}_ak8_{}_{}_{}_{}".format(self.process_name, var, jet, EVENT_SELECTION, lepton_selection), '', 
                             len(VARIABLES_BINNING[var])-1, array('d', VARIABLES_BINNING[var])), 
                            'selectedFatJets_nominal_{}_{}'.format(var, jet), 'weight'
                        )
                        output_histo.Write()
                        print('Total events --> {}'.format(output_histo.Integral()))

            # splitting signal samples considering gen matching with hadronic top from resonance: ==2 matched, ==1 matched, ==0 matched
            if self.is_sgn:
                print('\n### Splitting signal in 3 categories:')
                for filter, filter_label in zip(['==2', '==1', '==0'], ['2_jet_matched', '1_jet_matched', '0_jet_matched']):
                    print("Category: {}".format(filter_label))

                    root_df_jet_category_filtered = root_df_filtered.Filter(
                        "nselectedFatJets_nominal_matched_to_hadronic_top_from_resonance{}".format(filter), filter_label
                    )

                    for var in VARIABLES_BINNING.keys():
                        print('Variable --> {}'.format(var))
                        if var == 'invariant_mass_leading_subleading':
                            output_histo = root_df_jet_category_filtered.Histo1D(
                            ("{}_ak8_{}_{}_{}_{}_{}".format(self.process_name, var, EVENT_SELECTION, "2ak8", filter_label, lepton_selection), '', 
                                len(VARIABLES_BINNING[var])-1, array('d', VARIABLES_BINNING[var])), 
                            'selectedFatJets_nominal_{}'.format(var), 'weight'
                            ) 
                            output_histo.Write()
                            print('Total events --> {}'.format(output_histo.Integral()))
                        else: 
                            for ijet, jet in enumerate(['leading', 'subleading']): 
                                output_histo = root_df_jet_category_filtered.Histo1D(
                                    ("{}_ak8_{}_{}_{}_{}_{}_{}".format(self.process_name, var, jet, EVENT_SELECTION, "2ak8", filter_label, lepton_selection), '', 
                                        len(VARIABLES_BINNING[var])-1, array('d', VARIABLES_BINNING[var])), 
                                    'selectedFatJets_nominal_{}_{}'.format(var, jet), 'weight'
                                )
                                output_histo.Write()
                                print('Total events --> {}'.format(output_histo.Integral()))


    def _adding_new_columns(self, root_df):
        #### new columns definitions
        for boosted_jets in ['ak8', 'hotvr']:
            # ---ak4
            root_df = root_df.Define("selectedJets_nominal_outside_{}".format(boosted_jets),
                                     "selectedJets_nominal_is_inside_{}==0".format(boosted_jets))
            root_df = root_df.Define("nselectedJets_nominal_outside_{}".format(boosted_jets), 
                                     "Sum(selectedJets_nominal_outside_{})".format(boosted_jets))
            root_df = root_df.Define("selectedJets_nominal_outside_{}_btagDeepFlavB".format(boosted_jets), 
                                     "selectedJets_nominal_btagDeepFlavB[selectedJets_nominal_outside_{}]".format(boosted_jets))
            # ---

            # ---ak4 b-tagged
            root_df = root_df.Define("selectedBJets_nominal_outside_{}".format(boosted_jets), 
                                     "selectedJets_nominal_outside_{}_btagDeepFlavB>{}".format(boosted_jets, B_TAGGING_WP[str(self.year)]))
            root_df = root_df.Define("nselectedBJets_nominal_outside_{}".format(boosted_jets), 
                                     "Sum(selectedBJets_nominal_outside_{})".format(boosted_jets))
            # ---

        # ---boosted objects
        root_df = root_df.Define("selectedFatJets_nominal_top_tagged", 
                            "selectedFatJets_nominal_particleNet_TvsQCD >= {}".format(PNET_CUT[str(self.year)]))
        root_df = root_df.Define("nselectedFatJets_nominal_top_tagged", 
                                    "Sum(selectedFatJets_nominal_top_tagged)")

        for var in VARIABLES_BINNING.keys():
            if var == 'invariant_mass_leading_subleading':
                root_df = root_df.Filter(
                    "nselectedFatJets_nominal>=2", "2ak8" # necessary for invariant mass calculation
                ).Define(
                "selectedFatJets_nominal_{}".format(var),
                "invariant_mass(selectedFatJets_nominal_pt, selectedFatJets_nominal_eta, selectedFatJets_nominal_phi, selectedFatJets_nominal_mass)"
                ) 
            else:
                for ijet, jet in enumerate(['leading', 'subleading']): 
                    root_df = root_df.Define(
                    "selectedFatJets_nominal_{}_{}".format(var, jet),
                    "selectedFatJets_nominal_{}.size() > {} ? selectedFatJets_nominal_{}[{}] : -99".format(var, ijet, var, ijet)
                    ) # always checking the size of the array; if empty, segmentation violation error arises
        # ---

        # signal jet classification (matching with hadronic gen top from resonance)
        if self.is_sgn:
            root_df = root_df.Define("selectedFatJets_nominal_matched_to_hadronic_top_from_resonance", 
                                    "selectedFatJets_nominal_has_genTopHadronic_inside && selectedFatJets_nominal_has_genTopFromResonance_inside")
            # root_df = root_df.Define("selectedFatJets_nominal_not_matched_to_hadronic_top_from_resonance", 
            #                         "!(selectedFatJets_nominal_has_genTopHadronic_inside && selectedFatJets_nominal_has_genTopFromResonance_inside)")
            root_df = root_df.Define("nselectedFatJets_nominal_matched_to_hadronic_top_from_resonance", 
                                    "Sum(selectedFatJets_nominal_matched_to_hadronic_top_from_resonance)")
            
            # for var in VARIABLES_BINNING.keys():
            #     for matching in ['matched_to_hadronic_top_from_resonance', 'not_matched_to_hadronic_top_from_resonance']:
            #         root_df = root_df.Define(
            #             "selectedFatJets_nominal_{}_{}".format(matching, var),
            #             "selectedFatJets_nominal_{}[selectedFatJets_nominal_{}]".format(var, matching)
            #         )
            #         for ijet, jet in enumerate(['leading', 'subleading']): 
            #             root_df = root_df.Define(
            #             "selectedFatJets_nominal_{}_{}_{}".format(matching, var, jet),
            #             "selectedFatJets_nominal_{}_{}.size() > {} ? selectedFatJets_nominal_{}_{}[{}] : -99".format(matching, var, ijet, matching, var, ijet)
            #             ) # always checking the size of the array; if empty, segmentation violation error arises

        ####
        return root_df
    
    def _adding_event_selection(self, root_df_filtered, selection):
        root_df_filtered = root_df_filtered.Filter(
            selection, selection
        )
        return root_df_filtered

    def _event_selection(self, root_df, lepton_selection='ee', n_ak4_outside=2, n_b_outside=2, n_boosted_jets=1, boosted_jets='ak8'):

        root_df_filtered = root_df.Filter(
            "eventSelection_{}_cut".format(lepton_selection), "os_dilepton_selection"
        ).Filter(
            "nselectedJets_nominal_outside_{}>={}".format(boosted_jets, n_ak4_outside), "2ak4_outside"
        ).Filter(
            "nselectedBJets_nominal_outside_{}>={}".format(boosted_jets, n_b_outside), "2b_of_ak4_outside"
        )

        #just for BDT evaluation
        # root_df_filtered = root_df_filtered.Filter('run % 2 != 0', 'odd_events')

        if self.is_sgn and SIGNAL_ONLY_HADRONIC_TOP: root_df_filtered = root_df_filtered.Filter('hadronic_genTop_fromReso_filter(genTop_from_resonance, genTop_has_hadronically_decay)', 'hadronic_genTop_fromReso_events')

        return root_df_filtered



def main(input_file, output_dir, year, is_data, is_sgn, weighting, sys):
    # testing
    # input_file = "/nfs/dust/cms/user/gmilella/ttX_ntuplizer/sgn_2018_hotvr/merged/ttX_mass1250_width4_ntuplizer_output.root"
    # input_file = "/nfs/dust/cms/user/gmilella/ttX_ntuplizer/bkg_2018_hotvr/merged/tt_dilepton_MC2018_ntuplizer_7_merged.root"
    # output_dir = os.getcwd()

    processor = Processor(input_file, output_dir, year, is_data, is_sgn, weighting, sys)
    processor.process()


def parsing_file(file):
    global PROCESS_NAME

    print("Processing file: {}".format(file))

    # Convert file Path to string if it's a Path object
    file_str = file

    pattern = re.compile(r"merged\/(.*?)_MC")
    # Search for the pattern in case of background samples
    match = pattern.search(file_str)  # Use the string representation of the file path
    # Search for the pattern in case of background samples
    match = pattern.search(file_str)
    if match:
        print("Process name: {}".format(match.group(1)))
        PROCESS_NAME = match.group(1)
    else:
        # Search for the pattern in case of signal samples
        pattern = re.compile(r"merged\/(.*?)_ntuplizer")
        match = pattern.search(file_str)
        if match:
            print("Process name: {}".format(match.group(1)))
            PROCESS_NAME = match.group(1)
        else:
            print("No process name found.")
            sys.exit()

    return PROCESS_NAME

#################################################

def parse_args(argv=None):
    parser = ArgumentParser()

    parser.add_argument('--input_file', type=str, required=True,
        help="Input directory, where to find the h5 files")
    parser.add_argument('--output_dir', type=str,
        help="Top-level output directory. "
             "Will be created if not existing. "
             "If not provided, takes the input dir.")
    parser.add_argument('--year', type=int, required=True,
        help='Year of the samples.')
    parser.add_argument('--is_data', default=False, action='store_true',
        help='Flag when analyzing data samples')
    parser.add_argument('--is_sgn', default=False, action='store_true',
        help='Flag when analyzing signal samples')
    parser.add_argument('--weighting', default=False, action='store_true',
        help='If yes, normalization of the histogram per lumi, genweight and xsec')
    parser.add_argument('--sys', default=SYSTEMATIC,
        help='Systematic variations')

    args = parser.parse_args(argv)

    # If output directory is not provided, assume we want the output to be
    # alongside the input directory.
    if args.output_dir is None:
        args.output_dir = args.input_file

    # Return the options as a dictionary.
    return vars(args)

if __name__ == "__main__":
    args = parse_args()
    main(**args)
