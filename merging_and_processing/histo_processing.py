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
SCALING = {
    '500': 10,
    '1250': 20,
    '2000': 50
}
LINE_STYLE = {
    '500_4': 2, 
    '1250_4': 3, 
    '2000_4': 4,
}
REGION = 'SR1b2T'

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
        f'hotvr_scoreBDT_{jet_type}',
        # f'hotvr_mass_{jet_type}',
        # f'hotvr_pt_{jet_type}',
        # f'hotvr_eta_{jet_type}',
        # f'hotvr_phi_{jet_type}',
#         # f'hotvr_pt_vs_mass_{jet_type}',
        # f'hotvr_tau3_over_tau2_{jet_type}',
        # f'hotvr_fractional_subjet_pt_{jet_type}',
#         f'hotvr_scoreBDT_{jet_type}',
        # f'hotvr_min_pairwise_subjets_mass_{jet_type}',
        # f'hotvr_nsubjets_{jet_type}',
        # f'ak4_outside_hotvr_pt_{jet_type}', 
        # f'ak4_outside_hotvr_eta_{jet_type}', 
        # f'ak4_outside_hotvr_phi_{jet_type}', 
        # f'{jet_type}_pt',
        # f'{jet_type}_eta',
        # f'{jet_type}_phi',
        # f'{jet_type}_pfRelIso04_all',
    ])

REBINNING = True
IS_JET_COMPOSITION = False
INCLUDING_JET_FLAVORS = True
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

SIGNAL = '' 

class Processor:
    def __init__(self, input_dir, output_dir, year, weighting):
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.year = year
        self.weighting = weighting

        self.output_dir = self._creation_output_file()

    def _creation_output_file(self):
        # check if dir exist
        if self.input_dir == 'hotvr_variables': # and self.year != '2022' and self.year != '2022EE' and self.year != 'all_years_Run3':
            self.output_dir = os.path.join(self.output_dir, self.year, f"{self.input_dir}_{REGION}")
        else:
            self.output_dir = os.path.join(self.output_dir, self.year, f"{self.input_dir}_{REGION}")
        
        for variable in VARIABLES:
            if not os.path.exists(self.output_dir + '/' + variable):
                os.makedirs(self.output_dir + '/' + variable)
        return self.output_dir


    def process(self):
        global ADDITIONAL_FLAG

        # --- loading histos
        fpath = f'{self.year}/{self.input_dir}/{self.input_dir}_{EVENT_SELECTION}_bkg_data.root' #_OR_singleTrig
        if not EVENT_SELECTION:
            fpath = f'{self.year}/{self.input_dir}/{self.input_dir}.root'
        
        # if self.input_dir == 'hotvr_variables': #and self.year != '2022' and self.year != '2022EE' and self.year != 'all_years_Run3':
        #     fpath = f'{self.year}/{self.input_dir}/{self.input_dir}_{EVENT_SELECTION}.root'

        if not os.path.exists(fpath):
            print(f"File {fpath} does not exist.")
            sys.exit()
        # ---

        if 'cut_flow' in self.input_dir:
            ADDITIONAL_FLAG += '_weighted'

        print("Analyzing file: {}".format(fpath))

        for variable in VARIABLES:
            print(f'Variable: {variable}')
            output_filename = f"{self.output_dir}/{variable}/distributions"
            if SIGNAL:
                output_filename = f"{self.output_dir}/{variable}/distributions_{SIGNAL}"
            output_file = ROOT.TFile(f"{output_filename}.root", 'RECREATE')

            for systematic in utils.SYSTEMATICS:
                variations = ['Up', 'Down'] if systematic != 'nominal' else ['']

                for variation in variations:
                    print(f'\nSys: {systematic}{variation}')

                    for lepton_selection in LEPTON_SELECTIONS:
                        print(f"\nLepton {lepton_selection}")
                        dir_path = f'{systematic}{variation}/{lepton_selection}/'
                        if not EVENT_SELECTION:
                            dir_path = ''
                        if 'xSecTTbar' in systematic or 'xSecWJets' in dir_path:
                            dir_path = f'nominal/{lepton_selection}/'

                        # --- bkg processing
                        input_root_file = ROOT.TFile(fpath, "READ")
                        if systematic == 'PDF':
                            pdf_path = f'analysis_outputs/{self.year}/pdf_calculation_{REGION}/final_histos_{REGION}.root'
                            if SIGNAL == 'tta':
                                pdf_path = f'analysis_outputs/{self.year}/pdf_calculation_{REGION}/final_histos_{REGION}_tta.root'
                            if SIGNAL == 'tth':
                                pdf_path = f'analysis_outputs/{self.year}/pdf_calculation_{REGION}/final_histos_{REGION}_tth.root'
                            if os.path.exists(pdf_path):
                                input_root_file = ROOT.TFile(pdf_path, "READ")
                        
                        if IS_JET_COMPOSITION and systematic != 'PDF':
                            histos_bkg = utils.process_bkg_jet_composition(
                                input_root_file, 
                                dir_path, 
                                variable, 
                                additional_flag=ADDITIONAL_FLAG, 
                                rebinning=REBINNING, 
                                year=self.year,
                                systematic=f"{systematic}{variation}"
                            )
                        else:
                            histos_bkg = utils.process_bkg(
                                input_root_file,
                                dir_path,
                                variable, 
                                additional_flag=ADDITIONAL_FLAG,
                                rebinning=REBINNING,
                                year=self.year,
                                systematic=f"{systematic}{variation}"
                            )
                            
                        # --- data processing
                        if systematic == "nominal":
                            histo_data = utils.process_data(
                                input_root_file, 
                                dir_path, 
                                variable, 
                                additional_flag=ADDITIONAL_FLAG.replace('weighted', 'unweighted'), 
                                rebinning=REBINNING
                            )

                        # --- signal processing
                        if SIGNAL == 'tta' or SIGNAL == 'tth':
                            file_name_sgn = fpath.replace("bkg_data.root", "ttAorH.root")
                        elif SIGNAL == 'TZPrime':
                            file_name_sgn = fpath.replace("bkg_data.root", "tZprime.root")
                        else:
                            file_name_sgn = fpath.replace("bkg_data.root", "ttZprime.root")

                        if os.path.exists(file_name_sgn):
                            input_root_file = ROOT.TFile(file_name_sgn, "READ")

                        if systematic == 'PDF':
                            pdf_path = f'analysis_outputs/{self.year}/pdf_calculation_{REGION}/final_histos_{REGION}_ttZprime.root'
                            if SIGNAL == 'tta':
                                pdf_path = f'analysis_outputs/{self.year}/pdf_calculation_{REGION}/final_histos_{REGION}_tta.root'
                            if SIGNAL == 'tth':
                                pdf_path = f'analysis_outputs/{self.year}/pdf_calculation_{REGION}/final_histos_{REGION}_tth.root'

                            if not os.path.exists(pdf_path):
                                continue
                            input_root_file = ROOT.TFile(pdf_path, "READ")

                        histos_sgn = utils.process_sgn(
                            input_root_file, 
                            dir_path, 
                            variable, 
                            additional_flag=ADDITIONAL_FLAG, 
                            rebinning=REBINNING,
                            signal=SIGNAL
                        )

                        # --- creating output files directories and saving histos
                        output_directory = f'{systematic}{variation}/{lepton_selection}'
                        if systematic in utils.UNCORRELATED_SYSTEMATICS:
                            output_directory = f'{systematic}{variation}_{self.year}/{lepton_selection}'

                        output_file.cd()

                        if not output_file.GetDirectory(output_directory):
                            output_file.mkdir(output_directory)
                        output_file.cd(output_directory)

                        for mass_width, histo in histos_sgn.items():
                            if histo:
                                if isinstance(histo, ROOT.TH1):
                                    histo.SetDirectory(output_file)
                                    histo.Write()

                        if systematic == "nominal":
                            if histo_data:
                                # histo_data.SetBinContent(20, 1.26*histo_data.GetBinContent(20))
                                # histo_data.SetBinContent(19, 1.21*histo_data.GetBinContent(19))
                                histo_data.Write()
                        
                        for process, histo in histos_bkg.items():
                            if histo:
                                histo.Write()
                            # if systematic == "nominal":
                            #     print(f'Nominal MC tot: {histo.Integral()}')
                       # ---
            
                    # --- including jet flavors distributions
                    if INCLUDING_JET_FLAVORS:
                        input_root_file_flavors = ROOT.TFile(
                            fpath.replace(".root", "_jet_flavors.root"), "READ"
                        )
                        if systematic == 'PDF':
                            pdf_path = f'analysis_outputs/{self.year}/pdf_calculation_{REGION}/final_histos_{REGION}_jet_flavors.root'
                            if not os.path.exists(pdf_path):
                                continue
                            input_root_file_flavors = ROOT.TFile(pdf_path, "READ")

                        for lepton_selection in LEPTON_SELECTIONS:
                            dir_path = f'{systematic}{variation}/{lepton_selection}/'
                            if not EVENT_SELECTION:
                                dir_path = ''

                            for flavor in FLAVORS:
                                histos_bkg_flav = utils.process_bkg_jet_flavors(
                                    input_root_file_flavors,
                                    dir_path,
                                    variable,
                                    additional_flag=ADDITIONAL_FLAG,
                                    rebinning=REBINNING,
                                    year=self.year,
                                    flavor=flavor,
                                )

                                output_file.cd()
                                output_directory = f'{systematic}{variation}/{lepton_selection}'
                                if systematic in utils.UNCORRELATED_SYSTEMATICS:
                                    output_directory = f'{systematic}{variation}_{self.year}/{lepton_selection}'
                                if not EVENT_SELECTION:
                                    output_directory = ""
                                if output_directory and not output_file.GetDirectory(output_directory):
                                    output_file.mkdir(output_directory)
                                if output_directory:
                                    output_file.cd(output_directory)

                                for process, histo in histos_bkg_flav.items():
                                    if histo:
                                        histo.Write()
            # ---

            output_file.Close()
            input_root_file.Close()
            # ---

            print(f"Saving distributions in {output_file}.root")


def main(
        input_dir, output_dir, 
        year, weighting):

    print("Analyzing file: {}".format(input_dir))

    processor = Processor(input_dir, output_dir, year, weighting)
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
    parser.add_argument('--weighting', default=False, action='store_true',
        help='If yes, normalization of the histogram per lumi, genweight and xsec')

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
