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

LEPTON_SELECTION = ['single_muon'] 
# LEPTON_SELECTION = ['emu', 'mumu', 'ee'] 
ELECTRON_ID_TYPE = "MVA"
LEPTON_ID_MUON = "medium"
LEPTON_ID_ELE = "medium"
if LEPTON_SELECTION[0] == 'single_muon':
    LEPTON_ID_ELE = "loose"
print(f'Muon ID: {LEPTON_ID_MUON}, Electron ID: {LEPTON_ID_ELE}')

WEIGHTS_DICT = {
    'ee': "{} * {} * {} * {} * {}".format( 
        'trigger_weight_nominal',
        LEPTON_ID_ELE+"_"+ELECTRON_ID_TYPE+"_Electrons_weight_id_nominal", 
        LEPTON_ID_ELE+"_"+ELECTRON_ID_TYPE+"_Electrons_weight_recoPt_nominal",
        "tightRelIso_"+LEPTON_ID_MUON+"ID_Muons_weight_id_nominal",
          "tightRelIso_"+LEPTON_ID_MUON+"ID_Muons_weight_iso_nominal"),

    'emu': "{} * {} * {} * {} * {}".format(
        'trigger_weight_nominal',
        LEPTON_ID_ELE+"_"+ELECTRON_ID_TYPE+"_Electrons_weight_id_nominal", 
        LEPTON_ID_ELE+"_"+ELECTRON_ID_TYPE+"_Electrons_weight_recoPt_nominal",
        "tightRelIso_"+LEPTON_ID_MUON+"ID_Muons_weight_id_nominal",
          "tightRelIso_"+LEPTON_ID_MUON+"ID_Muons_weight_iso_nominal"),

    'mumu': "{} * {} * {} * {} * {}".format(
        'trigger_weight_nominal',
        LEPTON_ID_ELE+"_"+ELECTRON_ID_TYPE+"_Electrons_weight_id_nominal", 
        LEPTON_ID_ELE+"_"+ELECTRON_ID_TYPE+"_Electrons_weight_recoPt_nominal",
        "tightRelIso_"+LEPTON_ID_MUON+"ID_Muons_weight_id_nominal",
          "tightRelIso_"+LEPTON_ID_MUON+"ID_Muons_weight_iso_nominal"),

    'single_muon': "{} * {} * {} * {}".format(
        # 'trigger_weight_nominal',
        LEPTON_ID_ELE+"_"+ELECTRON_ID_TYPE+"_Electrons_weight_id_nominal", 
        LEPTON_ID_ELE+"_"+ELECTRON_ID_TYPE+"_Electrons_weight_recoPt_nominal",
        "tightRelIso_"+LEPTON_ID_MUON+"ID_Muons_weight_id_nominal",
          "tightRelIso_"+LEPTON_ID_MUON+"ID_Muons_weight_iso_nominal")
}
 
# VARIABLES =  ['invariant_mass_leading_subleading']
VARIABLES =  ['mass', 'scoreBDT', 'pt'] #'mass', 'scoreBDT']
VARIABLES =  ['tau3_over_tau2', 'nsubjets', 'min_pairwise_subjets_mass', 'fractional_subjet_pt']
# VARIABLES =  ['phi', 'eta', 'pt', 'mass', 'invariant_mass_leading_subleading', 'scoreBDT'] #'pt_vs_mass', 'pt', 'mass', 'invariant_mass_leading_subleading', 'nhotvr', 'ntagged_hotvr', 'ht_ak4_and_hotvr']
# VARIABLES = ['tau3_over_tau2', 'nsubjets', 'min_pairwise_subjets_mass', 'fractional_subjet_pt']

# VARIABLES =  ['pt_vs_mass', 'pt', 'mass', 'tau3_over_tau2', 'nsubjets', 'min_pairwise_subjets_mass', 'fractional_subjet_pt', 'scoreBDT', 'invariant_mass_leading_subleading', 'nhotvr', 'ntagged_hotvr', 'ht_ak4_and_hotvr'] #'max_eta_subjets', 'corrFactor', 'nhotvr']'tagged_pt', 
 #'max_eta_subjets']#['pt', 'mass'] #'invariant_mass_leading_subleading']#, 'pt', 'mass', 'ht_ak4_and_hotvr', 'deltaR']
#, 'mass', 'tau3_over_tau2', 'nsubjets', 'min_pairwise_subjets_mass', 'fractional_subjet_pt'] #'mass', 'invariant_mass_leading_subleading', 'ht_ak4_and_hotvr'] #, 'tau3_over_tau2', 'nsubjets', 'min_pairwise_subjets_mass']

VARIABLES_BINNING = OrderedDict()
VARIABLES_BINNING['pt_vs_mass'] = []
VARIABLES_BINNING['max_eta_subjets_VS_corrFactor'] = []
VARIABLES_BINNING['pt'] = list(np.arange(0., 3020., 20))
VARIABLES_BINNING['eta'] = list(np.arange(-2.5, 2.52, 0.02))
VARIABLES_BINNING['phi'] = list(np.arange(-3.5, 3.5, 0.02))
VARIABLES_BINNING['tagged_pt'] = list(np.arange(0., 3020., 20))
VARIABLES_BINNING['max_eta_subjets'] = list(np.arange(-3.5, 3.52, 0.02))
VARIABLES_BINNING['corrFactor'] = list(np.arange(.8, 1.5, 0.01))
VARIABLES_BINNING['mass'] = list(np.arange(0., 1510., 10))
VARIABLES_BINNING['ht_ak4_and_hotvr'] = list(np.arange(0., 5020., 20))
VARIABLES_BINNING['ht_ak4_outside_hotvr'] = list(np.arange(0., 5020., 20))
VARIABLES_BINNING['leadingVSsubleading_pt'] = list(np.arange(0., 6020., 20))
VARIABLES_BINNING['deltaR'] = list(np.arange(0., 5., 0.05))
VARIABLES_BINNING['leading_plus_subleading_pt'] = list(np.arange(0., 6020., 20))
VARIABLES_BINNING['leadingVSsubleading_pt'] = array('d', [200.0, 360.0, 440.0, 520.0, 700.0, 3000.0])
VARIABLES_BINNING['leadingVSsubleading_mass'] = list(np.arange(0., 1510., 10))
VARIABLES_BINNING['tau2_over_tau1'] = list(np.arange(0., 1.01, .01))
VARIABLES_BINNING['tau3_over_tau2'] = list(np.arange(0., 1.01, .01))
VARIABLES_BINNING['scoreBDT'] = list(np.arange(0., 1.005, .005))
VARIABLES_BINNING['fractional_subjet_pt'] = list(np.arange(0., 1.05, .05))
VARIABLES_BINNING['min_pairwise_subjets_mass'] = list(np.arange(0., 205., 5.))
VARIABLES_BINNING['nsubjets'] = list(np.arange(0., 5))
VARIABLES_BINNING['invariant_mass_leading_subleading'] = list(np.arange(0., 5020., 20))
VARIABLES_BINNING['nhotvr'] = list(np.arange(0., 6))
VARIABLES_BINNING['ntagged_hotvr'] = list(np.arange(0., 6))
VARIABLES_BINNING['MET_energy'] = list(np.arange(0., 3020., 20))

SYSTEMATIC_LABEL = 'nominal'
AK4_JEC_SYSTEMATIC = 'nominal'
BOOSTED_JEC_SYSTEMATIC = 'nominal'
print(f'Systematic label :{SYSTEMATIC_LABEL} -> AK4 sys {AK4_JEC_SYSTEMATIC}, HOTVR sys: {BOOSTED_JEC_SYSTEMATIC}')

BTAGGING_WP = 'loose'
print(f'bTagging WP: {BTAGGING_WP}')
NAK4, NBLOOSE = 2, 1
NBOOSTED_JETS = 2
BOOSTED_JETS = 'hotvr'
Z_PEAK_CUT = 'off' #'off' #'on'
NTAGGED_JETS = 1
if LEPTON_SELECTION[0] == 'single_muon':
    NAK4, NBLOOSE = 2, 1
    NBOOSTED_JETS = 1
    Z_PEAK_CUT = None

EVENT_SELECTION = f"after_2OS_{NAK4}ak4_outside_{BOOSTED_JETS}_{NBOOSTED_JETS}_{BOOSTED_JETS}"

if Z_PEAK_CUT == 'on':
    EVENT_SELECTION = "after_2OS_{}_Zpeak_{}_{}".format(Z_PEAK_CUT, NBOOSTED_JETS, BOOSTED_JETS) 
    LEPTON_SELECTION = ['ee', 'mumu'] #, 'emu']
    # EVENT_SELECTION = f"after_2OS_{Z_PEAK_CUT}_Zpeak_{NAK4}ak4_outside_{BOOSTED_JETS}_{NBOOSTED_JETS}_{BOOSTED_JETS}"
if Z_PEAK_CUT == 'off':
    EVENT_SELECTION = "after_2OS_{}_Zpeak_{}ak4_{}b_outside_{}_{}_{}".format(Z_PEAK_CUT, NAK4, NBLOOSE, BOOSTED_JETS, NBOOSTED_JETS, BOOSTED_JETS) 
    # EVENT_SELECTION = f"after_2OS_{Z_PEAK_CUT}_Zpeak_{NAK4}ak4_outside_{BOOSTED_JETS}_{NBOOSTED_JETS}_{BOOSTED_JETS}"
if LEPTON_SELECTION[0] == 'single_muon':
    EVENT_SELECTION = f"after_1mu_{NAK4}ak4_{NBLOOSE}b_outside_{BOOSTED_JETS}_{NBOOSTED_JETS}_{BOOSTED_JETS}"

JET_COMPOSITION_STUDY = False
if JET_COMPOSITION_STUDY: 
    # VARIABLES = ['mass', 'scoreBDT']
    VARIABLES = ['tau3_over_tau2', 'nsubjets', 'min_pairwise_subjets_mass', 'fractional_subjet_pt']

JET_FLAVOR = '' #'no_b_quark_and_no_hadronic_t' b_quark_or_hadronic_t   
BDT_WPs = [0.5]

Z_FIT_RESCALING = True
if LEPTON_SELECTION[0] == 'single_muon':
    Z_FIT_RESCALING = False
print(f'DY-rescaling: {Z_FIT_RESCALING}')

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

        if JET_COMPOSITION_STUDY:
            self.output_file, self.output_file_path = creation_output_file(
                input_file, output_dir, 
                "hotvr_variables_jet_composition", 
                year, 
                EVENT_SELECTION, 
                self.lepton_selection,
                self.systematic
            )
        else:
            self.output_file, self.output_file_path = creation_output_file(
                input_file, 
                output_dir, 
                "hotvr_variables", 
                year, 
                EVENT_SELECTION, 
                self.lepton_selection,
                self.systematic,
                miscellanea='_'+JET_FLAVOR
            )

    def process(self):
        root_df = ROOT.RDataFrame("Friends", str(self.input_file))
        if not self.is_data:
            print("Process: {}, XSec: {} pb, Sum of gen weights: {}".format(self.process_name, self.xsec, self.sum_gen_weights))
            weights = f"genweight * {self.xsec} * {LUMINOSITY[str(self.year)]} / {self.sum_gen_weights}" #
            root_df = root_df.Define("event_weight", weights)
            print(f'Event weight: {weights}')

        else: 
            print("Process: {}".format(self.process_name))

        print('EVENT SELECTION: {}'.format(EVENT_SELECTION))

        if self.is_data:
            if 'Muon_' in self.process_name and self.lepton_selection[0] == 'single_muon':
                self.lepton_selection = ['single_muon']
            elif 'DoubleLepton' in self.process_name or 'MuonEG' in self.process_name:
                self.lepton_selection = ['emu']
            elif 'DoubleEG' in self.process_name or 'EGamma' in self.process_name or 'Electron' in self.process_name: 
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

            if 'tbar' in self.process_name or 'tW' in self.process_name:
                  self.process_name = self.process_name.replace('dilepton', '2l')

        # adding new columns
        jet_composition = False
        if JET_COMPOSITION_STUDY or JET_FLAVOR:
            jet_composition = True
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
            BDT_SCORE_WPs=self.bdt_wps, 
            ntagged_jets=NTAGGED_JETS,
            is_jet_composition=jet_composition,
            nboosted_jets=NBOOSTED_JETS
        )
        print("Number of pre-selected events:", root_df.Count().GetValue())

        # event selection + histogram definitions
        for lepton_selection in self.lepton_selection:
            print('\nLepton Selection: {}'.format(lepton_selection))

            self.output_file.cd('{}/{}'.format(self.systematic, lepton_selection))

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
                process=self.process_name,
                is_data=self.is_data
            )
            print("Number of selected events:", root_df_filtered.Count().GetValue())

            if f'{NBOOSTED_JETS}_{BOOSTED_JETS}' in EVENT_SELECTION:
                selection_string = f"nselectedHOTVRJets_{self.boosted_systematic}=={NBOOSTED_JETS}"
                if NBOOSTED_JETS >= 1: 
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
                selection = f"selectedHOTVRJets_{self.boosted_systematic}_mass_leading < 300"
                root_df_filtered = adding_event_selection(
                    root_df_filtered,
                    selection
                )
                print(f"Additional filter: {selection_string}")

            # --- scale factors
            if not self.is_data:
                bweights = " btagSFlight_deepJet_L_"+self.year+" * btagSFbc_deepJet_L_"+self.year
                sf_weights = f"{bweights} * puWeight * bdt_sf_nominal * puID_L_sf"
                if self.year == '2022' or self.year == '2022EE':
                    sf_weights = f"{bweights} * puWeight * bdt_sf_nominal " # no PUId for Run 3 jets (all PUPPI)

                if self.year == '2017' or self.year == '2016' or self.year == '2016preVFP':
                    sf_weights += " * L1PreFiringWeight_Nom"
                if Z_FIT_RESCALING:
                    if 'dy_' in self.process_name:
                        sf_weights += f" * {self.z_fit_rescale[0]}"

                if "ME" in SYSTEMATIC_LABEL[0:2]:
                    if root_df.HasColumn("nLHEScaleWeight"):
                        nlhe = "XXX"
                        sf_weights += f" * LHEScaleWeight[{nlhe}]"
                        #normalization
                        sf_weights, nlhe = me_uncertainties_handler(root_df, sf_weights, nlhe)
                        if not math.isnan(self.sum_lhe_scale_weights[nlhe]):
                            sf_weights += f" / {self.sum_lhe_scale_weights[nlhe]}"
                if "ISR" in SYSTEMATIC_LABEL or "FSR" in SYSTEMATIC_LABEL:
                    sf_weights += " * PSWeight[XXX]"

                sf_weights += f" * {WEIGHTS_DICT[lepton_selection]}"
                if self.year == '2018':
                    sf_weights += " * (!HEM_veto ? (1.0 / 3.0) : 1.0)"

                # --- trigger SFs for Run 3 (N.B. they are not calculated in the ntuplization but in the utils!!!)
                if self.year == '2022' or self.year == '2022EE': 
                    sf_weights = sf_weights.replace("trigger_weight_nominal", f"trigger_sf_{lepton_selection}")
                # ---

                print(f"SF weights: {sf_weights}")

                # root_df_filtered = root_df_filtered.Define(
                #     "mass_window_weight",
                #     f"(selectedHOTVRJets_{self.boosted_systematic}_mass_leading > 150 && selectedHOTVRJets_{self.boosted_systematic}_mass_leading < 220) ? 0.9 : 1.0"
                # )

                root_df_filtered = root_df_filtered.Define("sf_weight", sf_weights)
                root_df_filtered = root_df_filtered.Define("weight", f"event_weight * {sf_weights}" )
                # root_df_filtered = root_df_filtered.Define("weight", f"event_weight * {sf_weights} * mass_window_weight" )
            # ---

            wp_str = str(BDT_WPs[0]).replace(".", "p")

            TAGGED_REGION_CATEGORIES = [f'ex{NTAGGED_JETS}tag', '2tag', 'less1tag'] #, '1tag'
            if lepton_selection == 'single_muon':
                TAGGED_REGION_CATEGORIES = ['1tag']
            TAGGED_REGION_CUTS = {
                'ex1tag':
                    f'((selectedHOTVRJets_{self.boosted_systematic}_is_top_tagged_wp_{wp_str}[0]) ^ '
                    f'(selectedHOTVRJets_{self.boosted_systematic}_is_top_tagged_wp_{wp_str}[1]))'
                , 
                '1tag':
                    f'(selectedHOTVRJets_{self.boosted_systematic}_is_top_tagged_wp_' 
                    f'{wp_str}[0] || '
                    f'selectedHOTVRJets_{self.boosted_systematic}_is_top_tagged_wp_' 
                    f'{wp_str}[1])'
                ,
                '2tag':
                    f'(selectedHOTVRJets_{self.boosted_systematic}_is_top_tagged_wp_' 
                    f'{wp_str}[0] && '
                    f'selectedHOTVRJets_{self.boosted_systematic}_is_top_tagged_wp_' 
                    f'{wp_str}[1])'
                ,
                'less1tag':
                    f'(selectedHOTVRJets_{self.boosted_systematic}_is_top_tagged_wp_' 
                    f'{wp_str}[0]==0 && '
                    f'selectedHOTVRJets_{self.boosted_systematic}_is_top_tagged_wp_' 
                    f'{wp_str}[1]==0)'
            }
            if lepton_selection == 'single_muon':
                TAGGED_REGION_CUTS = {
                    '1tag':
                    f'nselectedHOTVRJets_{self.boosted_systematic}_is_top_tagged_wp_{wp_str} >=1 '
                }

            # columns = root_df_filtered.AsNumpy([
            #     "bdt_sf_weight_bdt_nominal",
            #     "bdt_sf_nominal",
            #     "selectedHOTVRJets_nominal_pt"]
            # )
            # bdt1 = columns["bdt_sf_weight_bdt_nominal"]
            # pt_values = columns["selectedHOTVRJets_nominal_pt"]
            # bdt2 = columns["bdt_sf_nominal"]
            # for i, (pt_rvec, bdt11, bdt22) in enumerate(zip(pt_values, bdt1, bdt2)):
            #     if bdt11 < 1.0:
            #         print(f"Entry {i}:")
            #         print(f"  pt: {list(pt_rvec)}")
            #         print(f"  bdt11: {bdt11}")
            #         print(f" bdt22: {bdt22}")
            # sys.exit()

            for var in VARIABLES:
                print(f'\nVariable --> {var}')
                binning={
                    'bins_x': len(VARIABLES_BINNING[var]) - 1, 
                    'bins_x_edges': array('d', VARIABLES_BINNING[var]), 
                    'bins_y': [], 
                    'bins_y_edges': []}
                histo_var = {}
                histo_title = ''
                histo_type = 'TH1'

                if not JET_COMPOSITION_STUDY:
                    if var in ('invariant_mass_leading_subleading', 
                        'ht_ak4_and_hotvr', 
                        'ht_ak4_outside_hotvr', 
                        'deltaR', 
                        'leading_plus_subleading_pt', 
                        'nhotvr',
                        'ntagged_hotvr',
                        'MET_energy') or 'leadingVSsubleading' in var:
                        histo_var['var_x'] = f'selectedHOTVRJets_{self.boosted_systematic}_{var}'

                        histo_title = f"{self.process_name}_hotvr_{var}"
                        if JET_FLAVOR:
                            histo_title = f"{self.process_name}_hotvr_{var}_{JET_FLAVOR}"

                        if var == 'ht_ak4_and_hotvr': 
                            histo_var['var_x'] = 'ht_ak4_and_hotvr'
                        
                        if var == 'ht_ak4_and_hotvr': 
                            histo_var['var_x'] = 'ht_ak4_outside_hotvr'

                        if 'leadingVSsubleading' in var:
                            histo_type = 'TH2'
                            histo_var['var_x'] = f'selectedHOTVRJets_{self.boosted_systematic}_{var.split("_", 2)[1]}_leading'
                            histo_var['var_y'] = f'selectedHOTVRJets_{self.boosted_systematic}_{var.split("_", 2)[1]}_subleading'

                            binning['bins_y'] = len(VARIABLES_BINNING[var.split("_", 2)[1]]) - 1
                            binning['bins_y_edges'] =array('d', VARIABLES_BINNING[var.split("_", 2)[1]])

                        if 'nhotvr' == var:
                            histo_var['var_x'] = f'nselectedHOTVRJets_{self.boosted_systematic}'

                        if 'ntagged_hotvr' == var:
                            histo_var['var_x'] = f'nselectedHOTVRJets_{self.boosted_systematic}_is_top_tagged_wp_{wp_str}'

                        if var == 'MET_energy':
                            histo_var['var_x'] = var

                        additional_filter = ''
                        if JET_FLAVOR:
                            additional_filter = f"(selectedHOTVRJets_{self.boosted_systematic}_{JET_FLAVOR}[0]==1 || selectedHOTVRJets_{self.boosted_systematic}_{JET_FLAVOR}[1]==1)"
                            if JET_FLAVOR == 'pure_qcd':
                                additional_filter = f"(selectedHOTVRJets_{self.boosted_systematic}_{JET_FLAVOR}[0]==1 && selectedHOTVRJets_{self.boosted_systematic}_{JET_FLAVOR}[1]==1)"
                            if JET_FLAVOR == 'pure_qcd_or_hadronic_t':
                                additional_filter = (
                                    f"(selectedHOTVRJets_{self.boosted_systematic}_pure_qcd[0]==1 || "
                                    f"selectedHOTVRJets_{self.boosted_systematic}_pure_qcd[1]==1 || "
                                    f"selectedHOTVRJets_{self.boosted_systematic}_hadronic_t[0]==1 || "
                                    f"selectedHOTVRJets_{self.boosted_systematic}_hadronic_t[1]==1)"
                                )
                            if JET_FLAVOR == 'b_quark':
                                additional_filter = (
                                    f"(selectedHOTVRJets_{self.boosted_systematic}_b_from_top[0]==1 || "
                                    f"selectedHOTVRJets_{self.boosted_systematic}_b_not_from_top[0]==1) && "
                                    f"(selectedHOTVRJets_{self.boosted_systematic}_b_from_top[1]==1 || "
                                    f"selectedHOTVRJets_{self.boosted_systematic}_b_not_from_top[1]==1)"
                                )
                            if JET_FLAVOR == 'no_b_quark':
                                additional_filter = (
                                    f"(selectedHOTVRJets_{self.boosted_systematic}_b_from_top[0]==0 && "
                                    f"selectedHOTVRJets_{self.boosted_systematic}_b_not_from_top[0]==0) && "
                                    f"(selectedHOTVRJets_{self.boosted_systematic}_b_from_top[1]==0 && "
                                    f"selectedHOTVRJets_{self.boosted_systematic}_b_not_from_top[1]==0)"
                                )
                            if JET_FLAVOR == 'no_b_quark_and_no_hadronic_t':
                                additional_filter = (
                                    f"(selectedHOTVRJets_{self.boosted_systematic}_b_from_top[0]==0 && "
                                    f"selectedHOTVRJets_{self.boosted_systematic}_b_not_from_top[0]==0) && "
                                    f"(selectedHOTVRJets_{self.boosted_systematic}_b_from_top[1]==0 && "
                                    f"selectedHOTVRJets_{self.boosted_systematic}_b_not_from_top[1]==0) && "
                                    f"(selectedHOTVRJets_{self.boosted_systematic}_hadronic_t[0]==0 && "
                                    f"selectedHOTVRJets_{self.boosted_systematic}_hadronic_t[1]==0)"
                                )
                            if JET_FLAVOR == 'one_b_quark':
                                additional_filter = (
                                    f"(selectedHOTVRJets_{self.boosted_systematic}_b_from_top[0]==1 || "
                                    f"selectedHOTVRJets_{self.boosted_systematic}_b_not_from_top[0]==1) || "
                                    f"(selectedHOTVRJets_{self.boosted_systematic}_b_from_top[1]==1 || "
                                    f"selectedHOTVRJets_{self.boosted_systematic}_b_not_from_top[1]==1)"
                                )
                            if JET_FLAVOR == 'one_b_quark_and_no_hadronic_t':
                                additional_filter = (
                                    f"((selectedHOTVRJets_{self.boosted_systematic}_b_from_top[0]==1 || "
                                    f"selectedHOTVRJets_{self.boosted_systematic}_b_not_from_top[0]==1) || "
                                    f"(selectedHOTVRJets_{self.boosted_systematic}_b_from_top[1]==1 || "
                                    f"selectedHOTVRJets_{self.boosted_systematic}_b_not_from_top[1]==1)) && "
                                    f"(selectedHOTVRJets_{self.boosted_systematic}_hadronic_t[0]==0 && "
                                    f"selectedHOTVRJets_{self.boosted_systematic}_hadronic_t[1]==0)"
                                )
                            if JET_FLAVOR == 'b_quark_or_hadronic_t':
                                    additional_filter = (
                                        f"(selectedHOTVRJets_{self.boosted_systematic}_b_from_top[0]==1 || "
                                        f"selectedHOTVRJets_{self.boosted_systematic}_b_not_from_top[0]==1 || "
                                        f"selectedHOTVRJets_{self.boosted_systematic}_b_from_top[1]==1 || "
                                        f"selectedHOTVRJets_{self.boosted_systematic}_b_not_from_top[1]==1 || "
                                        f"selectedHOTVRJets_{self.boosted_systematic}_hadronic_t[0]==1 || "
                                        f"selectedHOTVRJets_{self.boosted_systematic}_hadronic_t[1]==1)"
                                    )
                            print(f'Additional filter (jet composition [{JET_FLAVOR}]) --> {additional_filter}')

                        # n_matched = root_df_filtered.Filter(additional_filter).Count().GetValue()
                        # print(f"[{JET_FLAVOR}] matched {n_matched} events with filter: {additional_filter}")

                        output_histo = histo_creation(
                            root_df_filtered, 
                            histo_title, 
                            histo_var, 
                            binning=binning, 
                            is_data=self.is_data, 
                            histo_type=histo_type, 
                            weight='weight',
                            additional_filter=additional_filter
                        )
                        self.output_file.cd(f'{self.systematic}/{lepton_selection}')
                        self.output_file.Get(f'{self.systematic}/{lepton_selection}').WriteObject(output_histo.GetPtr(), histo_title)
                        print('Total events: {}'.format(output_histo.Integral()))
                        output_histo.Delete()

                        for category in TAGGED_REGION_CATEGORIES:
                            print('Category: {}'.format(category))
                            histo_title = f"{self.process_name}_hotvr_{var}_{category}"
                            if JET_FLAVOR:
                                histo_title = f"{self.process_name}_hotvr_{var}_{category}_{JET_FLAVOR}"

                            # if 'nhotvr' == var:
                                # histo_title = f"{self.process_name}_{var}_{category}"
                                # histo_var['var_x'] = f'nselectedHOTVRJets_{self.boosted_systematic}_is_top_tagged_wp_{wp_str}'

                            # n_matched = root_df_filtered.Filter(additional_filter).Filter(TAGGED_REGION_CUTS[category], TAGGED_REGION_CUTS[category]).Count().GetValue()
                            # print(f"[{JET_FLAVOR}] matched {n_matched} events with filter: {additional_filter}")

                            output_histo = histo_creation(
                                root_df_filtered.Filter(TAGGED_REGION_CUTS[category], TAGGED_REGION_CUTS[category]), 
                                histo_title, 
                                histo_var, 
                                binning=binning, 
                                is_data=self.is_data, 
                                histo_type=histo_type, 
                                weight='weight',
                                additional_filter=additional_filter
                            )
                            self.output_file.cd(f'{self.systematic}/{lepton_selection}')
                            output_histo.Write()
                            print('Total events: {}'.format(output_histo.Integral()))
                            output_histo.Delete()
                        # ---

                    else: 
                        jet_list = ['leading', 'subleading']
                        if NBOOSTED_JETS == 1:
                            jet_list = ['leading']
                        for ijet, jet in enumerate(jet_list): 
                            histo_var['var_x'] = f'selectedHOTVRJets_{self.boosted_systematic}_{var}_{jet}'
                            print('Jet: {}'.format(jet))

                            if 'VS' in var:
                                histo_type = 'TH2'
                                histo_var['var_x'] = f'selectedHOTVRJets_{self.boosted_systematic}_corrFactor_{jet}'
                                histo_var['var_y'] = f'selectedHOTVRJets_{self.boosted_systematic}_max_eta_subjets_{jet}'

                                binning['bins_y'] = len(VARIABLES_BINNING['max_eta_subjets']) - 1
                                binning['bins_x'] = len(VARIABLES_BINNING['corrFactor']) - 1
                                binning['bins_y_edges'] =array('d', VARIABLES_BINNING['max_eta_subjets'])
                                binning['bins_x_edges'] =array('d', VARIABLES_BINNING['corrFactor'])

                            if var == 'pt_vs_mass':
                                histo_type = 'TH2'

                                histo_var['var_x'] = f'selectedHOTVRJets_{self.boosted_systematic}_pt_{jet}'
                                histo_var['var_y'] = f'selectedHOTVRJets_{self.boosted_systematic}_mass_{jet}'

                                binning['bins_y'] = len(VARIABLES_BINNING['mass']) - 1
                                binning['bins_x'] = len(VARIABLES_BINNING['pt']) - 1
                                binning['bins_y_edges'] =array('d', VARIABLES_BINNING['mass'])
                                binning['bins_x_edges'] =array('d', VARIABLES_BINNING['pt'])

                            if var == 'tagged_pt':
                                histo_var['var_x'] = f'selectedHOTVRJets_{self.boosted_systematic}_is_top_tagged_wp_{wp_str}_pt_{jet}'
                                binning['bins_x'] = len(VARIABLES_BINNING['pt']) - 1

                            histo_title = f"{self.process_name}_hotvr_{var}_{jet}"
                            additional_filter = ''
                            if JET_FLAVOR:
                                histo_title = f"{self.process_name}_hotvr_{var}_{jet}_{JET_FLAVOR}"
                                additional_filter = f"selectedHOTVRJets_{self.boosted_systematic}_{JET_FLAVOR}[{ijet}]==1"
                                print(f'Additional filter (jet composition [{JET_FLAVOR}]) --> {additional_filter}')

                            output_histo = histo_creation(
                                root_df_filtered, 
                                histo_title, 
                                histo_var, 
                                binning=binning, 
                                is_data=self.is_data, 
                                histo_type=histo_type, 
                                weight='weight', 
                                additional_filter=additional_filter
                            )
                            print('Total events: {}'.format(output_histo.Integral()))
                            
                            self.output_file.cd('{}/{}'.format(self.systematic, lepton_selection))
                            output_histo.Write()
                            output_histo.Delete()

                            # --- tagged jets
                            if var == 'scoreBDT': continue
                            if self.boosted_systematic == 'noJEC': continue

                            for category in TAGGED_REGION_CATEGORIES:
                                if var == 'tagged_pt': continue
                                print('Category: {}'.format(category))

                                if category == '1tag' or category == '2tag':
                                    histo_var['var_x'] = f'selectedHOTVRJets_{self.boosted_systematic}_is_top_tagged_wp_{wp_str}_{var}_{jet}'
                                    if var == 'pt_vs_mass':
                                        histo_var['var_x'] = f'selectedHOTVRJets_{self.boosted_systematic}_is_top_tagged_wp_{wp_str}_pt_{jet}'
                                        histo_var['var_y'] = f'selectedHOTVRJets_{self.boosted_systematic}_is_top_tagged_wp_{wp_str}_mass_{jet}'
                                else: 
                                    histo_var['var_x'] = f'selectedHOTVRJets_{self.boosted_systematic}_{var}_{jet}'
                                    if var == 'pt_vs_mass':
                                        histo_var['var_x'] = f'selectedHOTVRJets_{self.boosted_systematic}_pt_{jet}'
                                        histo_var['var_y'] = f'selectedHOTVRJets_{self.boosted_systematic}_mass_{jet}'

                                if var == 'pt_vs_mass':
                                    histo_type = 'TH2'

                                    binning['bins_y'] = len(VARIABLES_BINNING['mass']) - 1
                                    binning['bins_x'] = len(VARIABLES_BINNING['pt']) - 1
                                    binning['bins_y_edges'] =array('d', VARIABLES_BINNING['mass'])
                                    binning['bins_x_edges'] =array('d', VARIABLES_BINNING['pt'])

                                histo_title = f"{self.process_name}_hotvr_{var}_{jet}_{category}"
                                if JET_FLAVOR:
                                    histo_title = f"{self.process_name}_hotvr_{var}_{jet}_{category}_{JET_FLAVOR}"

                                output_histo = histo_creation(
                                    root_df_filtered.Filter(TAGGED_REGION_CUTS[category], TAGGED_REGION_CUTS[category]), 
                                    histo_title, 
                                    histo_var, 
                                    binning=binning, 
                                    is_data=self.is_data, 
                                    histo_type=histo_type, 
                                    weight='weight',
                                    additional_filter=additional_filter
                                )
                                self.output_file.cd('{}/{}'.format(self.systematic, lepton_selection))
                                output_histo.Write()
                                print('Total events: {}'.format(output_histo.Integral()))
                                output_histo.Delete()
                            # ---

                else:
                    print('JET COMPOSITION STUDY')
                    if self.is_data: 
                        print('Jet composition studies not possible on data! [Change the flag!!]')
                        sys.exit()

                    for jet_composition in JET_COMPOSITION_FLAGS_MERGED.keys():
                        print('\nJET TYPE: {}'.format(jet_composition))

                        jet_list = ['leading', 'subleading']
                        if NBOOSTED_JETS == 1:
                            jet_list = ['leading']
                        
                        for ijet, jet in enumerate(jet_list): 
                            print('Jet: {}'.format(jet))

                            histo_title = f"{self.process_name}_hotvr_{jet_composition}_{var}_{jet}"
                            histo_var['var_x'] = 'selectedHOTVRJets_{}_{}_{}_{}'.format(
                                self.boosted_systematic, 
                                jet_composition, 
                                var, 
                                jet
                            )

                            output_histo = histo_creation(
                                root_df_filtered, 
                                histo_title, 
                                histo_var, 
                                binning=binning, 
                                is_data=self.is_data, 
                                histo_type=histo_type, 
                                weight='weight'
                            )
                            self.output_file.cd('{}/{}'.format(self.systematic, lepton_selection))
                            self.output_file.Get(f'{self.systematic}/{lepton_selection}').WriteObject(output_histo.GetPtr(), histo_title)
                            output_histo.Write()
                            print('Total events: {}'.format(output_histo.Integral()))
                            output_histo.Delete()

                            if var == 'scoreBDT': continue

                            for category in TAGGED_REGION_CATEGORIES:
                                print('Category: {}'.format(category))

                                histo_var['var_x'] = f'selectedHOTVRJets_{self.boosted_systematic}_{jet_composition}_{var}_{jet}'

                                histo_title = f"{self.process_name}_hotvr_{jet_composition}_{var}_{jet}_{category}"

                                output_histo = histo_creation(
                                    root_df_filtered.Filter(TAGGED_REGION_CUTS[category], TAGGED_REGION_CUTS[category]), 
                                    histo_title, 
                                    histo_var, 
                                    binning=binning, 
                                    is_data=self.is_data, 
                                    histo_type=histo_type, 
                                    weight='weight'
                                )
                                self.output_file.cd('{}/{}'.format(self.systematic, lepton_selection))
                                output_histo.Write()
                                print('Total events: {}'.format(output_histo.Integral()))
                                output_histo.Delete()

            del root_df_filtered


def main(input_files, output_dir, year, is_data, is_sgn, weighting, systematic, ak4_systematic, boosted_systematic):
    # testing
    # output_dir = os.getcwd()

    # processor = Processor(input_file, output_dir, year, is_data, is_sgn, weighting, systematic, ak4_systematic, boosted_systematic)
    # processor.process()

    for input_file in input_files:
        # input_file = "/data/dust/user/gmilella/ttX_ntuplizer/data_2022EE_hotvr/merged/Muon_2022EE_E_1_merged.root"
        # input_file = "/data/dust/user/gmilella/ttX_ntuplizer/data_2018_hotvr/merged/DoubleEG_2018_D_1_merged.root"
        # input_file = "/data/dust/user/gmilella/ttX_ntuplizer/data_2018_hotvr/merged/DoubleMuon_2018_D_1_merged.root"
        # input_file = '/data/dust/user/gmilella/ttX_ntuplizer/sgn_2022_central_hotvr/merged/TTZprimeToTT_M-1500_Width10_1_merged.root'
        # input_file = "/data/dust/user/gmilella/ttX_ntuplizer/bkg_2018_hotvr/merged/tt_semilepton_MC2018_ntuplizer_3_merged.root"
        # input_file = "/data/dust/user/gmilella/ttX_ntuplizer/sgn_2018_central_hotvr/merged/TTZprimeToTT_M-1500_Width4_1_merged.root"
        # input_file = "/data/dust/user/gmilella/ttX_ntuplizer/bkg_2016_hotvr/merged/dy_ht_2500_MC2016_ntuplizer_3_merged.root"
        # input_file = "/data/dust/user/gmilella/ttX_ntuplizer/sgn_2016preVFP_central_hotvr/merged/TTZprimeToTT_M-500_Width4_merged.root"
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
