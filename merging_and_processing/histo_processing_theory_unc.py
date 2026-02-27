import os, sys

from argparse import ArgumentParser
from collections import OrderedDict

import ROOT
from array import array

ROOT.ROOT.EnableImplicitMT()
ROOT.gROOT.SetBatch(True)

import utils_root_canvas as utils

LEPTON_SELECTIONS = ['ee', 'mumu', 'emu']
# LEPTON_SELECTIONS = ['single_muon']
SGN_MASSES = ['500_4', '750_4', '1000_4', '1250_4', '1500_4', '1750_4', '2000_4', '2500_4', '3000_4', '4000_4']
MASSES = ['500', '750', '1000', '1250', '1500', '1750', '2000', '2500', '3000', '4000']
WIDTHS = ['4', '10', '20', '50']

REGION = 'CR2J2T'

if 'SR2b' in REGION:
    EVENT_SELECTION = 'after_2OS_off_Zpeak_2ak4_2b_outside_hotvr_2_hotvr'
if 'SR1b' in REGION:
    EVENT_SELECTION = 'after_2OS_off_Zpeak_2ak4_1b_outside_hotvr_2_hotvr'
if 'CR' in REGION:
    EVENT_SELECTION = 'after_2OS_on_Zpeak_2_hotvr'
    LEPTON_SELECTIONS = ['ee', 'mumu']
if 'CR2J_noZrequirement' in REGION:
    EVENT_SELECTION = 'after_2OS_2_hotvr'
if 'CR_' in REGION:
    EVENT_SELECTION = 'after_2OS'
if 'CR_onZ' in REGION:
    EVENT_SELECTION = 'after_2OS_on_Zpeak'
if 'single_muon' in REGION:
    EVENT_SELECTION = 'after_1mu_2ak4_1b_outside_hotvr_1_hotvr'
    LEPTON_SELECTIONS = ['single_muon']
if REGION == 'SR':
    EVENT_SELECTION = 'after_2OS_off_Zpeak'
if not REGION:
    EVENT_SELECTION = ''

ADDITIONAL_FLAG = ''
if '2T' in REGION:
    ADDITIONAL_FLAG = '_2tag'
if '1T' in REGION and 'ex' not in REGION:
    ADDITIONAL_FLAG = '_1tag'
if 'ex' in REGION:
    ADDITIONAL_FLAG = '_ex1tag'
if '0T' in REGION:
    ADDITIONAL_FLAG = '_less1tag'
if 'T' not in REGION:
    ADDITIONAL_FLAG = ''

VARIABLES = [
    # 'cutflow'
    'hotvr_invariant_mass_leading_subleading',
    # 'nhotvr',
    # 'ntagged_hotvr',
    # 'ht_ak4_and_hotvr',
    # 'eta_vs_phi'
    # 'dilepton_invariant_mass_leading_subleading',
    # 'nelectrons', 
    # 'nmuons',
    # 'met_and_muon_pt'
    # 'PV_npvsGood',
    # 'Nb_outside_vs_Ntop',
    # "hotvr_MET_energy"
    # "hadronic_gentops_pt",
    # f"gentops_from_resonance_pt",
    # f"hadronic_gentops_from_resonance_pt",
    # 'mjj_resonance',
]

for jet_type in ['leading']: #, 'subleading']: #, 'subleading']:
    VARIABLES.extend([
        # f'hotvr_scoreBDT_{jet_type}',
        # f'hotvr_mass_{jet_type}',
        # f'hotvr_pt_{jet_type}',
        # f'hotvr_eta_{jet_type}',
        # f'hotvr_phi_{jet_type}',
    ])

FLAVORS = [
    'hadronic_t', 
    # 'pure_qcd_or_hadronic_t', 
    # 'pure_qcd',
    # 'b_quark',
    # 'one_b_quark',
    # 'one_b_quark_and_no_hadronic_t',
    # 'no_b_quark', 
    # 'no_b_quark_and_no_hadronic_t',
    # 'b_quark_or_hadronic_t'
]

THEORY_SPLITTING = ['TT', 'TTX', 'MultiTop', 'V', 'ST']
THEORY_UNCERTAINTIES = ['FSR', 'ISR', 'MEenv', 'MEfac', 'MEren']
MAPPING = {
    'TT': 'tt',
    'TTX': 'ttX',
    'MultiTop': 'multitop',
    'V': 'dy',
    'ST': 'ST'
}

SIGNAL = '' 

class Processor:
    def __init__(self, input_dir, output_dir, year):
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.year = year

        self.output_dir = self._creation_output_file()

    def _creation_output_file(self):
        # check if dir exist        
        for variable in VARIABLES:
            self.output_dir = f"{self.input_dir}/{self.year}/hotvr_variables_{REGION}/{variable}/"
            if not os.path.exists(self.output_dir):
                print(f"Output directory {self.output_dir} does not exist")
                sys.exit()
        return self.output_dir

    def process(self):
        global ADDITIONAL_FLAG

        # --- output
        output_filename = f"{self.output_dir}/distributions_theory_unc"
        print(f"Saving distributions in: {output_filename}.root")
        output_file = ROOT.TFile(f'{output_filename}.root', 'RECREATE') 

        # --- loading histos
        for variable in VARIABLES:
            print(f'Variable: {variable}')
            fpath = f"{self.input_dir}/{self.year}/hotvr_variables_{REGION}/{variable}/distributions.root"

            if not os.path.exists(fpath):
                print(f"File {fpath} does not exist.")
                sys.exit()

            print(f'Reading from {fpath}')
            input_file = ROOT.TFile(fpath, 'READ')

            for systematic in THEORY_UNCERTAINTIES:
                variations = ['Up', 'Down'] if systematic != 'nominal' else ['']
                
                for variation in variations:
                    # print(f'\nSys: {systematic}{variation}')
                    
                    for theory_process in THEORY_SPLITTING:
                        output_dir = f"{systematic}{theory_process}{variation}"
                        output_file.mkdir(output_dir)

                        for lepton_selection in LEPTON_SELECTIONS:
                            # print(f"\nLepton {lepton_selection}")

                            output_file.mkdir(f"{output_dir}/{lepton_selection}")

                            # --- background
                            h_var = input_file.Get(f"{systematic}{variation}/{lepton_selection}/{MAPPING[theory_process]}_{variable}")
                            if not h_var: 
                                continue
                            h_tot = h_var.Clone(f'tot_bkg_{variable}')
                            
                            output_file.cd(f"{output_dir}/{lepton_selection}")
                            h_var.Write()

                            if theory_process == 'V':
                                h_wjets = input_file.Get(f"{systematic}{variation}/{lepton_selection}/wjets_{variable}")
                                if h_wjets:
                                    h_tot.Add(h_wjets)
            
                                    output_file.cd(f"{output_dir}/{lepton_selection}")
                                    h_wjets.Write()
                            
                            for process in utils.PROCESSES:
                                if 'tot_bkg' in process or f'{MAPPING[theory_process]}' == process:
                                    continue
                                if theory_process == 'V' and 'wjets' in process:
                                    continue

                                h = input_file.Get(f"nominal/{lepton_selection}/{process}_{variable}")
                                if not h: 
                                    continue
                                # print(theory_process, h.GetName())

                                output_file.cd(f"{output_dir}/{lepton_selection}")
                                h.Write()

                                h_tot.Add(h)
        
                            output_file.cd(f"{output_dir}/{lepton_selection}")
                            h_tot.Write()

                            # --- background (by flavor)
                            h_var = input_file.Get(
                                f"{systematic}{variation}/{lepton_selection}/{MAPPING[theory_process]}_hadronic_t_{variable}"
                            )
                            if not h_var:
                                continue
                            h_tot = h_var.Clone(f'tot_bkg_hadronic_t_{variable}')
                            
                            output_file.cd(f"{output_dir}/{lepton_selection}")
                            h_var.Write()

                            if theory_process == 'V':
                                h_wjets = input_file.Get(f"{systematic}{variation}/{lepton_selection}/wjets_hadronic_t_{variable}")
                                if h_wjets:
                                    h_tot.Add(h_wjets)
            
                                    output_file.cd(f"{output_dir}/{lepton_selection}")
                                    h_wjets.Write()
                            
                            for process in utils.PROCESSES:
                                if 'tot_bkg' in process or f'{MAPPING[theory_process]}' == process:
                                    continue
                                if theory_process == 'V' and 'wjets' in process:
                                    continue

                                h = input_file.Get(f"nominal/{lepton_selection}/{process}_hadronic_t_{variable}")
                                if not h: 
                                    continue

                                output_file.cd(f"{output_dir}/{lepton_selection}")
                                h.Write()

                                h_tot.Add(h)
        
                            output_file.cd(f"{output_dir}/{lepton_selection}")
                            h_tot.Write()
                            

def main(
        input_dir, output_dir, year
        ):

    print("Analyzing file: {}".format(input_dir))

    processor = Processor(input_dir, output_dir, year)
    processor.process()


#################################################

def parse_args(argv=None):
    parser = ArgumentParser()

    parser.add_argument('--input_dir', type=str, required=True,
        help="Input file, where to find the h5 files")
    parser.add_argument('--output_dir', type=str, required=True,
        help="Top-level output directory. "
             "Will be created if not existing. "
             "If not provided, takes the input dir.")
    parser.add_argument('--year', type=str, required=True,
        help='Year of the samples.')

    args = parser.parse_args(argv)

    # If output directory is not provided, assume we want the output to be
    # alongside the input directory.
    if args.input_dir is None: 
        args.input_dir = os.getcwd()
    if args.output_dir is None:
        args.output_dir = args.input_dir

    # Return the options as a dictionary.
    return vars(args)

if __name__ == "__main__":
    args = parse_args()
    main(**args)
