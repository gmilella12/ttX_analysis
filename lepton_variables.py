import os, sys
import re
from argparse import ArgumentParser
from itertools import product

import math

import yaml
from yaml.loader import SafeLoader

import numpy as np
from array import array

from collections import OrderedDict

sys.path.append(os.getcwd())
from utils_folder.utils import *

import ROOT
ROOT.ROOT.EnableImplicitMT()

ROOT_DIR = os.getcwd()

LUMINOSITY = {
    '2018': 59830, '2017': 41480,
    '2016preVFP': 19500, '2016': 16500,
    '2022': 7875, '2022EE': 27000 
}

LEPTON_SELECTION = ['single_muon'] #['ee', 'mumu', 'emu'] #'single_muon'] #]
ELECTRON_ID_TYPE = "MVA"
LEPTON_ID_MUON = "medium"
LEPTON_ID_ELE = "loose"
print(f'Muon ID: {LEPTON_ID_MUON}, Electron ID: {LEPTON_ID_ELE}')

WEIGHTS_DICT = {
    'ee': "event_weight * {} * {} * {} * {} * {}".format( 
        'trigger_weight_nominal',
        LEPTON_ID_ELE+"_"+ELECTRON_ID_TYPE+"_Electrons_weight_id_nominal", 
        LEPTON_ID_ELE+"_"+ELECTRON_ID_TYPE+"_Electrons_weight_recoPt_nominal",
        "tightRelIso_"+LEPTON_ID_MUON+"ID_Muons_weight_id_nominal",
          "tightRelIso_"+LEPTON_ID_MUON+"ID_Muons_weight_iso_nominal"),

    'emu': "event_weight * {} * {} * {} * {} * {}".format(
        'trigger_weight_nominal',
        LEPTON_ID_ELE+"_"+ELECTRON_ID_TYPE+"_Electrons_weight_id_nominal", 
        LEPTON_ID_ELE+"_"+ELECTRON_ID_TYPE+"_Electrons_weight_recoPt_nominal",
        "tightRelIso_"+LEPTON_ID_MUON+"ID_Muons_weight_id_nominal",
          "tightRelIso_"+LEPTON_ID_MUON+"ID_Muons_weight_iso_nominal"),

    'mumu': "event_weight * {} * {} * {} * {} * {}".format(
        'trigger_weight_nominal',
        LEPTON_ID_ELE+"_"+ELECTRON_ID_TYPE+"_Electrons_weight_id_nominal", 
        LEPTON_ID_ELE+"_"+ELECTRON_ID_TYPE+"_Electrons_weight_recoPt_nominal",
        "tightRelIso_"+LEPTON_ID_MUON+"ID_Muons_weight_id_nominal",
          "tightRelIso_"+LEPTON_ID_MUON+"ID_Muons_weight_iso_nominal"),

    'single_muon': "event_weight * {} * {} * {} * {}".format(
        # 'trigger_weight_nominal',
        LEPTON_ID_ELE+"_"+ELECTRON_ID_TYPE+"_Electrons_weight_id_nominal", 
        LEPTON_ID_ELE+"_"+ELECTRON_ID_TYPE+"_Electrons_weight_recoPt_nominal",
        "tightRelIso_"+LEPTON_ID_MUON+"ID_Muons_weight_id_nominal",
          "tightRelIso_"+LEPTON_ID_MUON+"ID_Muons_weight_iso_nominal")
}

SYSTEMATIC_LABEL = 'nominal'
AK4_JEC_SYSTEMATIC = 'nominal'
BOOSTED_JEC_SYSTEMATIC = 'nominal'

VARIABLES = ['pt', 'phi', 'eta', 'met_and_muon_pt', 'nmuons', 'nelectrons', 'delta_phi_bjet_muon', 'delta_phi_hotvr_muon'] #'invariant_mass_leading_subleading'] #, 'pt', 'phi', 'eta', 'charge']
VARIABLES_BINNING = OrderedDict()
VARIABLES_BINNING['pt'] = list(np.arange(0., 2005., 5))
VARIABLES_BINNING['met_and_muon_pt'] = list(np.arange(0., 2005., 5))
VARIABLES_BINNING['phi'] = list(np.arange(-3.5, 3.5, 0.02))
VARIABLES_BINNING['delta_phi_bjet_muon'] = list(np.arange(0, 7, 0.02))
VARIABLES_BINNING['delta_phi_hotvr_muon'] = list(np.arange(0, 7, 0.02))
VARIABLES_BINNING['eta'] = list(np.arange(-2.5, 2.52, 0.02))
VARIABLES_BINNING['invariant_mass_leading_subleading'] = list(np.arange(0., 1005., 1.))
VARIABLES_BINNING['charge'] = list(range(-1, 3))
VARIABLES_BINNING['nmuons'] = list(range(0, 10))
VARIABLES_BINNING['nelectrons'] = list(range(0, 10))

BTAGGING_WP = 'loose'
print(f'bTagging WP: {BTAGGING_WP}')
NAK4, NBLOOSE = 1, 1
NBOOSTED_JETS = 1
BOOSTED_JETS = 'hotvr'
Z_PEAK_CUT = None
NTAGGED_JETS = 1

EVENT_SELECTION = f"after_2OS_{NAK4}ak4_outside_{BOOSTED_JETS}_{NBOOSTED_JETS}_{BOOSTED_JETS}"
if Z_PEAK_CUT == 'on':
    EVENT_SELECTION = "after_2OS_{}_Zpeak_{}_{}".format(Z_PEAK_CUT, NBOOSTED_JETS, BOOSTED_JETS) 
    LEPTON_SELECTION = ['ee', 'mumu', 'emu'] #, 'emu']
    # EVENT_SELECTION = f"after_2OS_{Z_PEAK_CUT}_Zpeak_{NAK4}ak4_outside_{BOOSTED_JETS}_{NBOOSTED_JETS}_{BOOSTED_JETS}"
if Z_PEAK_CUT == 'off':
    EVENT_SELECTION = "after_2OS_{}_Zpeak_{}ak4_{}b_outside_{}_{}_{}".format(Z_PEAK_CUT, NAK4, NBLOOSE, BOOSTED_JETS, NBOOSTED_JETS, BOOSTED_JETS) 
    # EVENT_SELECTION = f"after_2OS_{Z_PEAK_CUT}_Zpeak_{NAK4}ak4_outside_{BOOSTED_JETS}_{NBOOSTED_JETS}_{BOOSTED_JETS}"
if LEPTON_SELECTION[0] == 'single_muon':
    EVENT_SELECTION = f"after_1mu_{NAK4}ak4_outside_{BOOSTED_JETS}_{NBOOSTED_JETS}_{BOOSTED_JETS}"

BDT_WPs = [0.5]

Z_FIT_RESCALING = False

class Processor:
    def __init__(self, input_file, output_dir, year, is_data, is_sgn, weighting, systematic, ak4_systematic, boosted_systematic):
        self.input_file = input_file
        self.output_dir = output_dir
        self.year = year
        self.is_data = is_data
        self.is_sgn = is_sgn
        self.weighting = weighting
        self.systematic = systematic
        self.ak4_systematic = ak4_systematic
        self.boosted_systematic = boosted_systematic
        self.lepton_selection = LEPTON_SELECTION 
        self.bdt_wps = BDT_WPs

        self.process_name = parsing_file(input_file)
        if not self.is_data:
            self.xsec = xsec(self.process_name, is_sgn, year=year)
            self.sum_gen_weights = sum_gen_weights(input_file, self.process_name, is_sgn, year)
            self.sum_lhe_scale_weights = sum_lhe_scale_weights(input_file, self.process_name, is_sgn, year)

            if Z_FIT_RESCALING:
                self.z_fit_rescale = Z_peak_fit_results(year)

        self.output_file, self.output_file_path = creation_output_file(
            input_file, 
            output_dir, 
            "lepton_variables", 
            year, 
            EVENT_SELECTION, 
            self.lepton_selection,
            self.systematic
        ) 


    def process(self):
        root_df = ROOT.RDataFrame("Friends", str(self.input_file))
        if not self.is_data:
            print("Process: {}, XSec: {} pb, Sum of gen weights: {}".format(self.process_name, self.xsec, self.sum_gen_weights))
            weights = f"genweight * puWeight * {self.xsec} * {LUMINOSITY[str(self.year)]} / {self.sum_gen_weights}"
            bweights = "btagSFlight_deepJet_L_"+self.year+" * btagSFbc_deepJet_L_"+self.year

            tot_weights = f"{bweights} * {weights}"
            if self.year == '2017' or self.year == '2016' or self.year == '2016preVFP':
                tot_weights += " * L1PreFiringWeight_Nom"
            if Z_FIT_RESCALING:
                if 'dy_' in self.process_name:
                    tot_weights += f" * {self.z_fit_rescale[0]}"

            if "ME" in SYSTEMATIC_LABEL[0:2]:
                if root_df.HasColumn("nLHEScaleWeight"):
                    nlhe = 1
                    if not math.isnan(self.sum_lhe_scale_weights[nlhe]):
                        tot_weights += f" * LHEScaleWeight[{nlhe}]"
                        #normalization
                        tot_weights, nlhe = me_uncertainties_handler(root_df, tot_weights, nlhe)
                        tot_weights += f" / {self.sum_lhe_scale_weights[nlhe]}"
            if "ISR" in SYSTEMATIC_LABEL or "FSR" in SYSTEMATIC_LABEL:
                tot_weights += " * PSWeight[XXX]"
            print(f"Tot. weights: {tot_weights}")

            root_df = root_df.Define("event_weight", tot_weights)
        else: 
            print("Process: {}".format(self.process_name))

        print('EVENT SELECTION: {}'.format(EVENT_SELECTION))

        if self.is_data:
            if 'Muon_' in self.process_name and self.lepton_selection[0] == 'single_muon':
                self.lepton_selection = ['single_muon']
            elif 'DoubleLepton' in self.process_name or 'MuonEG' in self.process_name:
                self.lepton_selection = ['emu']
            elif 'DoubleEG' in self.process_name or 'EGamma' in self.process_name:
                self.lepton_selection = ['ee']
            elif 'Muon_' in self.process_name:
                self.lepton_selection = ['mumu']

        # additional cut on LHE_HT for inclusive DY m50 LO
        if self.process_name == 'dy_m-50' or self.process_name == 'dy_to2L_m-50':
            if self.year == '2022' or self.year == '2022EE':
                root_df = adding_event_selection(root_df, "LHE_Njets == 0")
            else:
                root_df = adding_event_selection(root_df, "LHE_HT < 70")
        # additional cut on LHE_Njets for NLO DYJ(->ll)
        if re.search(r'_(\d+J)', self.process_name):
            lhe_njets = int(re.search(r'_(\d+J)', self.process_name).group(1)[0])
            if lhe_njets == 2:
                root_df = adding_event_selection(root_df, "LHE_Njets >= {}".format(lhe_njets))
            else:
                root_df = adding_event_selection(root_df, "LHE_Njets == {}".format(lhe_njets))
        # additional cut for jetVeto map
        if self.year == '2022' or self.year == '2022EE':
            root_df = adding_event_selection(root_df, "jetMapVeto==0")


        # adding new columns
        root_df = adding_new_columns(
            root_df, 
            LEPTON_ID_ELE=LEPTON_ID_ELE, 
            ELECTRON_ID_TYPE=ELECTRON_ID_TYPE, 
            LEPTON_ID_MUON=LEPTON_ID_MUON,
            year=self.year, 
            ak4_systematic=self.ak4_systematic, 
            boosted_systematic=self.boosted_systematic, 
            process=self.process_name, 
            is_data=self.is_data,
            BDT_SCORE_WPs=self.bdt_wps
        )

        # event selection + histogram definitions
        for lepton_selection in self.lepton_selection:
            print('\nLepton Selection: {}'.format(lepton_selection))

            root_df_filtered = event_selection(
                root_df, 
                lepton_selection=lepton_selection, 
                n_ak4_outside=NAK4, 
                n_b_outside=NBLOOSE, 
                LEPTON_ID_ELE=LEPTON_ID_ELE, 
                ELECTRON_ID_TYPE=ELECTRON_ID_TYPE, 
                LEPTON_ID_MUON=LEPTON_ID_MUON,
                boosted_jets=BOOSTED_JETS, 
                on_Z_peak=Z_PEAK_CUT, 
                ak4_systematic=self.ak4_systematic, 
                boosted_systematic=self.boosted_systematic, 
                year=self.year, 
                process=self.process_name
            )

            if lepton_selection == 'single_muon':
                selections = [
                    f"is_BJets_{self.ak4_systematic}_{BTAGGING_WP}_outside_{BOOSTED_JETS}_and_tightRelIso_{LEPTON_ID_MUON}ID_Muons_in_same_hemisphere", 
                    f"is_selectedHOTVRJets_{self.boosted_systematic}_and_tightRelIso_{LEPTON_ID_MUON}ID_Muons_in_same_hemisphere == 0"
                    f"tightRelIso_{LEPTON_ID_MUON}ID_Muons_pt[0] > 50 && MET_energy > 300"
                ]
                for selection_string in selections:
                    root_df_filtered = adding_event_selection(
                        root_df_filtered, 
                        selection_string
                    )

            # dxy, dz cuts on electrons
            # root_df_filtered = adding_event_selection(root_df_filtered,
            #                                                 "dxy_dz_electron_cut({}_{}_Electrons_dxy, {}_{}_Electrons_dz, {}_{}_Electrons_eta)".format(
            #                                                     LEPTON_ID_ELE, ELECTRON_ID_TYPE, LEPTON_ID_ELE, ELECTRON_ID_TYPE, LEPTON_ID_ELE, ELECTRON_ID_TYPE))

            if f'{NBOOSTED_JETS}_{BOOSTED_JETS}' in EVENT_SELECTION:
                selection_string = f"nselectedHOTVRJets_{self.boosted_systematic}=={NBOOSTED_JETS}"
                if NBOOSTED_JETS >= 2: 
                     selection_string = f"nselectedHOTVRJets_{self.boosted_systematic}>={NBOOSTED_JETS}"
                
                root_df_filtered = adding_event_selection(
                    root_df_filtered, 
                    selection_string
                )
                # ---filtering out the over-corrected jets
                selection = f"ROOT::VecOps::All(selectedHOTVRJets_{self.boosted_systematic}_max_eta_subjets < 2.4 && selectedHOTVRJets_{self.boosted_systematic}_max_eta_subjets > -2.4)"
                root_df_filtered = adding_event_selection(
                    root_df_filtered,
                    selection
                )

            if not self.is_data:
                if self.is_sgn:
                    WEIGHTS_DICT[lepton_selection] += " * bdt_sf_weight_bdt_nominal"

                weight_formula = WEIGHTS_DICT[lepton_selection]
                if self.year == '2018':
                    weight_formula += " * (!HEM_veto ? (1.0 / 3.0) : 1.0)"

                root_df_filtered = root_df_filtered.Define("weight", weight_formula)
                # ---

            for var in VARIABLES:
                print('Variable --> {}'.format(var))

                binning={
                    'bins_x': len(VARIABLES_BINNING[var]) - 1, 
                    'bins_x_edges': array('d', VARIABLES_BINNING[var]), 
                    'bins_y': [], 
                    'bins_y_edges': []
                }
                
                histo_var = {}
                histo_title = f"{self.process_name}_{var}"
                histo_type = 'TH1'

                variable_mappings = {
                    'invariant_mass_leading_subleading': f'dilepton_invariant_{lepton_selection}_mass',
                    'met_and_muon_pt': 'met_and_muon_pt',
                    'nmuons': f'ntightRelIso_{LEPTON_ID_MUON}ID_Muons',
                    'nelectrons': f'n{LEPTON_ID_ELE}_{ELECTRON_ID_TYPE}_Electrons',
                    'delta_phi_bjet_muon': f"delta_phi_BJets_{self.ak4_systematic}_{BTAGGING_WP}_outside_{BOOSTED_JETS}_and_tightRelIso_{LEPTON_ID_MUON}ID_Muons",
                    'delta_phi_hotvr_muon': f"delta_phi_selectedHOTVRJets_{self.boosted_systematic}_and_tightRelIso_{LEPTON_ID_MUON}ID_Muons",
                }

                if var in variable_mappings:
                    histo_var['var_x'] = variable_mappings[var]
                    if var == 'invariant_mass_leading_subleading':
                        histo_title = f"{self.process_name}_dilepton_{var}"
                
                    output_histo = histo_creation(
                            root_df_filtered, 
                            histo_title, 
                            histo_var, 
                            binning=binning, 
                            is_data=self.is_data, 
                            histo_type=histo_type, 
                            weight='weight'
                    )
                    self.output_file.cd(f'{self.systematic}/{lepton_selection}')
                    output_histo.Write()
                    print('Total events: {}'.format(output_histo.Integral()))
                    del output_histo

                else:
                    if lepton_selection == 'emu':
                        for lepton, lepton_label in zip(
                            ['electron', 'muon'],
                            [f'{LEPTON_ID_ELE}_{ELECTRON_ID_TYPE}_Electrons_{var}_leading', f'tightRelIso_{LEPTON_ID_MUON}ID_Muons_{var}_leading']
                        ):
                            histo_title = f"{self.process_name}_{lepton}_{var}"
                            histo_var['var_x'] = lepton_label
                            print(lepton_label)

                            output_histo = histo_creation(
                                root_df_filtered, 
                                histo_title, 
                                histo_var, 
                                binning=binning, 
                                is_data=self.is_data, 
                                histo_type=histo_type, 
                                weight='weight'
                            )
                            self.output_file.cd(f'{self.systematic}/{lepton_selection}')
                            output_histo.Write()
                            print('Total events: {}'.format(output_histo.Integral()))
                            del output_histo
                            continue
                    else:
                        jet_multiplicity = ['leading', 'subleading'] if lepton_selection != 'single_muon' else ['leading']

                        for ijet, jet in enumerate(jet_multiplicity):
                            histo_title = f"{self.process_name}_{jet}_{var}"
                            if lepton_selection == 'ee':
                                histo_var['var_x'] = f'{LEPTON_ID_ELE}_{ELECTRON_ID_TYPE}_Electrons_{var}_{jet}'
                            elif lepton_selection == 'mumu' or lepton_selection == 'single_muon':
                                if (var == 'dz' or var == 'dxy'): continue
                                histo_var['var_x'] = f'tightRelIso_{LEPTON_ID_MUON}ID_Muons_{var}_{jet}'
                            else: continue

                            output_histo = histo_creation(
                                root_df_filtered, 
                                histo_title, 
                                histo_var, 
                                binning=binning, 
                                is_data=self.is_data, 
                                histo_type=histo_type, 
                                weight='weight'
                            )
                            self.output_file.cd(f'{self.systematic}/{lepton_selection}')
                            output_histo.Write()
                            print('Total events: {}'.format(output_histo.Integral()))
                            del output_histo


def main(input_files, output_dir, year, is_data, is_sgn, weighting, systematic, ak4_systematic, boosted_systematic):

    for input_file in input_files:
        # testing
        # input_file = "/data/dust/user/gmilella/ttX_ntuplizer/bkg_2016_hotvr/merged/ttH_HToNonbb_MC2018_ntuplizer_merged.root"
        # input_file = "/data/dust/user/gmilella/ttX_ntuplizer/bkg_2016preVFP_hotvr/merged/dy_ht_1200_MC2016preVFP_ntuplizer_1_merged.root"
        # input_file = "/data/dust/user/gmilella/ttX_ntuplizer/bkg_2018_hotvr/merged/dy_ht_1200_MC2018_ntuplizer_1_merged.root"
        # input_file = "/data/dust/user/gmilella/ttX_ntuplizer/sgn_2018_central_hotvr/merged/TTZprimeToTT_M-1500_Width4_1_merged.root"
        # input_file = "/data/dust/user/gmilella/ttX_ntuplizer/data_2018_hotvr/merged/SingleMuon_2018_D_8_merged.root"
        # input_file = "/data/dust/user/gmilella/ttX_ntuplizer/data_2016_hotvr/merged/tt_dilepton_MC2022_ntuplizer_10_merged.root"
        # input_file = "/data/dust/user/gmilella/ttX_ntuplizer/bkg_2022_hotvr/merged/tt_dilepton_MC2022_ntuplizer_1_merged.root" 

        print(f"Starting processing for file: {input_file}")
        processor = Processor(input_file, output_dir, year, is_data, is_sgn, weighting, systematic, ak4_systematic, boosted_systematic)
        processor.process()


#################################################

def parse_args(argv=None):
    parser = ArgumentParser()

    parser.add_argument('--input_files', type=str, nargs='+', required=True, help="List of input files to process")
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
    parser.add_argument('--systematic', default=SYSTEMATIC_LABEL,
        help='Systematic variations')
    parser.add_argument('--ak4_systematic', default=AK4_JEC_SYSTEMATIC,
        help='JEC variations')
    parser.add_argument('--boosted_systematic', default=BOOSTED_JEC_SYSTEMATIC,
        help='JEC variations')

    args = parser.parse_args(argv)

    # If output directory is not provided, assume we want the output to be
    # alongside the input directory.
    if args.output_dir is None:
        args.output_dir = os.path.dirname(args.input_files[0])

    # Return the options as a dictionary.
    return vars(args)

if __name__ == "__main__":
    args = parse_args()
    main(**args)
