import os, sys
import re
from argparse import ArgumentParser
from itertools import product

import yaml
from yaml.loader import SafeLoader

import numpy as np
from array import array

from collections import OrderedDict

sys.path.append('/afs/desy.de/user/g/gmilella/ttX3_post_ntuplization_analysis/ttX_analysis')
from utils_folder.utils import *

import ROOT
ROOT.ROOT.EnableImplicitMT()

ROOT_DIR = '/afs/desy.de/user/g/gmilella/ttX3_post_ntuplization_analysis/ttX_analysis'
cpp_functions_header = "{}/cpp_functions_header.h".format(ROOT_DIR)
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
LEPTON_ID_MUON = "loose"
LEPTON_ID_ELE = "medium"

WP = [0.87] 
WP_CUTS_LABELS = {
    (0.87, 0.87): 'lead_loose_OR_sublead_loose',
    (0.87): 'sublead_loose'
}
B_TAGGING_WP = {
    '2016preVFP': 
        {'loose': 0.0614, 'medium': 0.3093, 'tight': 0.7221}, #https://btv-wiki.docs.cern.ch/ScaleFactors/UL2016preVFP/
    '2016': 
        {'loose': 0.0480, 'medium': 0.2489, 'tight': 0.6377}, #https://btv-wiki.docs.cern.ch/ScaleFactors/UL2016postVFP/
    '2017': 
        {'loose': 0.0532, 'medium': 0.3040, 'tight': 0.7476}, #https://btv-wiki.docs.cern.ch/ScaleFactors/UL2017/
    '2018': 
        {'loose': 0.0490, 'medium': 0.2783, 'tight': 0.7100}, #https://btv-wiki.docs.cern.ch/ScaleFactors/UL2018/
} 
PNET_CUT = {
    '2018': '0.58', '2017': '0.58',
    '2016': '0.5', '2016preVFP': '0.5'
}

WEIGHTS_DICT = {
    'ee': "event_weight * {} * {} * {} ".format(
        'trigger_weight_nominal', 
        LEPTON_ID_ELE+"_"+ELECTRON_ID_TYPE+"_Electrons_weight_id_nominal", 
        LEPTON_ID_ELE+"_"+ELECTRON_ID_TYPE+"_Electrons_weight_recoPt_nominal"),

    'emu': "event_weight * {} * {} * {} * {} * {}".format(
        'trigger_weight_nominal',
        LEPTON_ID_ELE+"_"+ELECTRON_ID_TYPE+"_Electrons_weight_id_nominal", 
        LEPTON_ID_ELE+"_"+ELECTRON_ID_TYPE+"_Electrons_weight_recoPt_nominal",
        "tightRelIso_"+LEPTON_ID_MUON+"ID_Muons_weight_id_nominal",
          "tightRelIso_"+LEPTON_ID_MUON+"ID_Muons_weight_iso_nominal"),

    'mumu': "event_weight * {} * {} * {}".format(
        'trigger_weight_nominal', 
        "tightRelIso_"+LEPTON_ID_MUON+"ID_Muons_weight_id_nominal", 
        "tightRelIso_"+LEPTON_ID_MUON+"ID_Muons_weight_iso_nominal")}

SYSTEMATIC = 'nominal'

VARIABLES_BINNING = OrderedDict()
# {
#     'scoreBDT': list(np.arange(0., 1.005, .005)), 'pt': list(np.arange(0., 3020., 20)), 'mass': list(np.arange(0., 1510., 10)),  
#     'tau2_over_tau1': list(np.arange(0., 1.01, .01)), 
# } #'min_deltaR_vs_ak4': list(np.arange(0., 10.2, 0.2)), 'min_deltaR_vs_lepton': list(np.arange(0., 10.2, 0.2))
VARIABLES_BINNING['pt'] = list(np.arange(0., 3020., 20))
VARIABLES_BINNING['mass'] = list(np.arange(0., 1510., 10))
# VARIABLES_BINNING['tau2_over_tau1'] = list(np.arange(0., 1.01, .01))
VARIABLES_BINNING['tau3_over_tau2'] = list(np.arange(0., 1.01, .01))
# VARIABLES_BINNING['scoreBDT'] = list(np.arange(0., 1.005, .005))
VARIABLES_BINNING['fractional_subjet_pt'] = list(np.arange(0., 1.05, .05))
VARIABLES_BINNING['min_pairwise_subjets_mass'] = list(np.arange(0., 205., 5.))
VARIABLES_BINNING['nsubjets'] = list(np.arange(0., 5))
# VARIABLES_BINNING['invariant_mass_leading_subleading'] = list(np.arange(0., 5020., 20))

BOOSTED_JETS = 'hotvr'
Z_PEAK_CUT = 'off'
Z_PEAK_LOW_EDGE, Z_PEAK_HIGH_EDGE = 80., 101.
EVENT_SELECTION = "after_2OS_{}_Zpeak_2ak4_2b_outside_{}".format(Z_PEAK_CUT, BOOSTED_JETS)

SIGNAL_ONLY_HADRONIC_TOP = False

class Processor:
    def __init__(self, input_file, output_dir, year, is_data, is_sgn, weighting, sys):
        self.input_file = input_file
        self.output_dir = output_dir
        self.year = year
        self.is_data = is_data
        self.is_sgn = is_sgn
        self.weighting = weighting
        self.sys = sys  
        self.lepton_selection = LEPTON_SELECTION

        self.process_name = parsing_file(input_file)
        if not self.is_data:
            self.xsec = xsec(self.process_name, is_sgn)
            self.sum_gen_weights = sum_gen_weights(input_file, self.process_name, is_sgn, year)
        self.output_file = creation_output_file(input_file, output_dir, "hotvr_variables", year, EVENT_SELECTION, self.sys) 

    def process(self):
        root_df = ROOT.RDataFrame("Friends", str(self.input_file))
        if not self.is_data:
            print("Process: {}, XSec: {} pb, Sum of gen weights: {}".format(self.process_name, self.xsec, self.sum_gen_weights))
            root_df = root_df.Define("event_weight", 
                                     "genweight * puWeight * btagSFlight_deepJet_L_{} * btagSFbc_deepJet_L_{} * {} * {} / {}".format(
                                          str(self.year), str(self.year), self.xsec, LUMINOSITY[str(self.year)], self.sum_gen_weights))
        else: 
            print("Process: {}".format(self.process_name))

        if self.is_data:
            if 'DoubleLepton' in self.process_name or 'MuonEG' in self.process_name:
                self.lepton_selection = ['emu']
            elif 'DoubleEG' in self.process_name or 'EGamma' in self.process_name:
                self.lepton_selection = ['ee']
            elif 'DoubleMuon' in self.process_name:
                self.lepton_selection = ['mumu']

        # adding new columns
        root_df = self._adding_new_columns(root_df)

        # event selection + histogram definitions
        for lepton_selection in self.lepton_selection:
            print('\nLepton Selection: {}'.format(lepton_selection))
            
            root_df_filtered = self._event_selection(root_df, lepton_selection=lepton_selection, n_ak4_outside=2, n_b_outside=2, boosted_jets=BOOSTED_JETS)
            if not self.is_data:
                root_df_filtered = root_df_filtered.Define("weight", WEIGHTS_DICT[lepton_selection])

            self.output_file.cd(lepton_selection)

            if Z_PEAK_CUT == 'on':
                root_df_filtered = self._adding_event_selection(root_df_filtered,
                                                                 "dilepton_invariant_{}_mass>={} & dilepton_invariant_{}_mass <= {}".format(lepton_selection, Z_PEAK_LOW_EDGE, lepton_selection, Z_PEAK_HIGH_EDGE))
            if Z_PEAK_CUT == 'off':
                root_df_filtered = self._adding_event_selection(root_df_filtered, 
                                                                "dilepton_invariant_{}_mass<{} || dilepton_invariant_{}_mass > {}".format(lepton_selection, Z_PEAK_LOW_EDGE, lepton_selection, Z_PEAK_HIGH_EDGE))

            for var in VARIABLES_BINNING.keys():
                print('Variable --> {}'.format(var))

                bins = len(VARIABLES_BINNING[var]) - 1
                bin_edges = array('d', VARIABLES_BINNING[var])
                histo_var, histo_title = '', ''

                if var == 'invariant_mass_leading_subleading':
                    histo_title = "{}_hotvr_{}_{}_{}".format(self.process_name, var, EVENT_SELECTION, lepton_selection)
                    histo_var = 'preselectedHOTVRJets_{}'.format(var)

                    if not self.is_data:
                        output_histo = root_df_filtered.Histo1D((histo_title, '', bins, bin_edges), histo_var, 'weight')
                    else:
                        output_histo = root_df_filtered.Histo1D((histo_title, '', bins, bin_edges), histo_var)
                    output_histo.Write()
                    print('Total events: {}'.format(output_histo.Integral()))

                    # histo_title = "{}_hotvr_top_tagged_cut_based_{}_{}_{}".format(self.process_name, var, EVENT_SELECTION, lepton_selection)
                    # histo_var = 'preselectedHOTVRJets_top_tagged_cut_based_{}'.format(var)

                    if not self.is_data:
                        output_histo = root_df_filtered.Histo1D((histo_title, '', bins, bin_edges), histo_var, 'weight')
                    else:
                        output_histo = root_df_filtered.Histo1D((histo_title, '', bins, bin_edges), histo_var)
                    output_histo.Write()
                    print('Total events: {}'.format(output_histo.Integral()))
                
                else: 
                    for ijet, jet in enumerate(['leading', 'subleading']): 
                        histo_title = "{}_hotvr_{}_{}_{}_{}".format(self.process_name, var, jet, EVENT_SELECTION, lepton_selection)
                        histo_var = 'preselectedHOTVRJets_{}_{}'.format(var, jet)

                        if not self.is_data:
                            output_histo = root_df_filtered.Histo1D((histo_title, '', bins, bin_edges), histo_var, 'weight')
                        else:
                            output_histo = root_df_filtered.Histo1D((histo_title, '', bins, bin_edges), histo_var)
                        output_histo.Write()
                        print('Total events: {}'.format(output_histo.Integral()))


    def _adding_new_columns(self, root_df):
        #### new columns definitions
        # ---invariant di-lepton mass
        root_df = root_df.Define(
            "dilepton_invariant_ee_mass",
            "invariant_mass({}_{}_Electrons_pt, {}_{}_Electrons_eta, {}_{}_Electrons_phi, {}_{}_Electrons_mass)".format(
                LEPTON_ID_ELE, ELECTRON_ID_TYPE, LEPTON_ID_ELE, ELECTRON_ID_TYPE, LEPTON_ID_ELE, ELECTRON_ID_TYPE, LEPTON_ID_ELE, ELECTRON_ID_TYPE
            )
        )
        root_df = root_df.Define(
            "dilepton_invariant_mumu_mass",
            "invariant_mass(tightRelIso_{}ID_Muons_pt, tightRelIso_{}ID_Muons_eta, tightRelIso_{}ID_Muons_phi, tightRelIso_{}ID_Muons_mass)".format(
                LEPTON_ID_MUON, LEPTON_ID_MUON, LEPTON_ID_MUON, LEPTON_ID_MUON
            )
        )
        root_df = root_df.Define(
            "dilepton_invariant_emu_mass",
            "invariant_mass_emu({}_{}_Electrons_pt, {}_{}_Electrons_eta, {}_{}_Electrons_phi, {}_{}_Electrons_mass,"
            "tightRelIso_{}ID_Muons_pt, tightRelIso_{}ID_Muons_eta, tightRelIso_{}ID_Muons_phi, tightRelIso_{}ID_Muons_mass)".format(
                LEPTON_ID_ELE, ELECTRON_ID_TYPE, LEPTON_ID_ELE, ELECTRON_ID_TYPE, LEPTON_ID_ELE, ELECTRON_ID_TYPE, LEPTON_ID_ELE, ELECTRON_ID_TYPE, 
                LEPTON_ID_MUON, LEPTON_ID_MUON, LEPTON_ID_MUON, LEPTON_ID_MUON
            )
        )
        # ---

        # ---b-jets
        for WP in B_TAGGING_WP[str(self.year)].keys():
            if root_df.HasColumn("nselectedBJets_nominal_{}".format(WP)): continue
            root_df = root_df.Define(
                "selectedBJets_nominal_{}".format(WP), 
                "selectedJets_nominal_btagDeepFlavB>{}".format(B_TAGGING_WP[str(self.year)][WP])
            )
            root_df = root_df.Define(
                "nselectedBJets_nominal_{}".format(WP),
                "Sum(selectedBJets_nominal_{})".format(WP)
                )
        # ---

        for boosted_jets, boosted_jets_label in zip(['ak8', 'hotvr'], ['selectedFatJets_nominal', 'preselectedHOTVRJets']):
            # ---ak4
            root_df = root_df.Define("selectedJets_nominal_outside_{}".format(boosted_jets),
                                     "selectedJets_nominal_is_inside_{}==0".format(boosted_jets))
            root_df = root_df.Define("nselectedJets_nominal_outside_{}".format(boosted_jets), 
                                     "Sum(selectedJets_nominal_outside_{})".format(boosted_jets))
            root_df = root_df.Define("selectedJets_nominal_outside_{}_btagDeepFlavB".format(boosted_jets), 
                                     "selectedJets_nominal_btagDeepFlavB[selectedJets_nominal_outside_{}]".format(boosted_jets))
            # ---

            # ---ak4 b-tagged
            for WP in B_TAGGING_WP[str(self.year)].keys():
                root_df = root_df.Define("selectedBJets_nominal_{}_outside_{}".format(WP, boosted_jets), 
                                        "selectedJets_nominal_outside_{}_btagDeepFlavB>{}".format(boosted_jets, B_TAGGING_WP[str(self.year)][WP]))
                root_df = root_df.Define("nselectedBJets_nominal_{}_outside_{}".format(WP, boosted_jets), 
                                        "Sum(selectedBJets_nominal_{}_outside_{})".format(WP, boosted_jets))
            # ---

        # ---boosted objects
        for var in VARIABLES_BINNING.keys():
            if var == 'invariant_mass_leading_subleading':
                root_df = root_df.Define(
                "preselectedHOTVRJets_{}".format(var),
                "invariant_mass(preselectedHOTVRJets_pt, preselectedHOTVRJets_eta, preselectedHOTVRJets_phi, preselectedHOTVRJets_mass)"
                ) 
            else:
                for ijet, jet in enumerate(['leading', 'subleading']): 
                    root_df = root_df.Define(
                    "preselectedHOTVRJets_{}_{}".format(var, jet),
                    "preselectedHOTVRJets_{}.size() > {} ? preselectedHOTVRJets_{}[{}] : -99".format(var, ijet, var, ijet)
                    ) # always checking the size of the array; if empty, segmentation violation error arises
        # ---

        ####
        return root_df
    
    def _adding_event_selection(self, root_df_filtered, selection):
        root_df_filtered = root_df_filtered.Filter(
            selection, selection
        )
        return root_df_filtered

    def _event_selection(self, root_df, lepton_selection='ee', n_ak4_outside=2, n_b_outside=2, b_wp='loose', boosted_jets='hotvr'):

        root_df_filtered = root_df.Filter(
            "eventSelection_{}_cut".format(lepton_selection), "os_dilepton_selection"
        ).Filter(
            "nselectedJets_nominal_outside_{}>={}".format(boosted_jets, n_ak4_outside), "2ak4_outside_{}".format(boosted_jets)
        ).Filter(
            "nselectedBJets_nominal_{}_outside_{}>={}".format(b_wp, boosted_jets, n_b_outside), "2b_of_ak4_outside_{}".format(boosted_jets)
        )

        if self.is_sgn and SIGNAL_ONLY_HADRONIC_TOP: root_df_filtered = root_df_filtered.Filter('hadronic_genTop_fromReso_filter(genTop_from_resonance, genTop_has_hadronically_decay)', 'hadronic_genTop_fromReso_events')

        return root_df_filtered


def main(input_file, output_dir, year, is_data, is_sgn, weighting, sys):
    # testing
    # input_file = "/nfs/dust/cms/user/gmilella/ttX_ntuplizer/sgn_2018_hotvr/merged/ttX_mass1250_width4_ntuplizer_output.root"
    # input_file = "/nfs/dust/cms/user/gmilella/ttX_ntuplizer/sgn_2018_central_hotvr/merged/TTZprimeToTT_M-750_Width4_merged.root"
    # input_file = "/nfs/dust/cms/user/gmilella/ttX_ntuplizer/bkg_2018_hotvr/merged/tt_dilepton_MC2018_ntuplizer_7_merged.root"
    # input_file = "/nfs/dust/cms/user/gmilella/ttX_ntuplizer/data_2018_hotvr/merged/DoubleLepton_2018_C_merged.root"
    # input_file = "/nfs/dust/cms/user/gmilella/ttX_ntuplizer/data_2016preVFP_hotvr/merged/DoubleMuon_2016preVFP_D_5_merged.root"
    # output_dir = os.getcwd()

    processor = Processor(input_file, output_dir, year, is_data, is_sgn, weighting, sys)
    processor.process()



#################################################

def parse_args(argv=None):
    parser = ArgumentParser()

    parser.add_argument('--input_file', type=str, required=True,
        help="Input directory, where to find the h5 files")
    parser.add_argument('--output_dir', type=str,
        help="Top-level output directory. "
             "Will be created if not existing. "
             "If not provided, takes the input dir.")
    parser.add_argument('--year', type=str, required=True,
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
