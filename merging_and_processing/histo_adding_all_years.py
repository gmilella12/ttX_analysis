import os, sys, re

import glob, subprocess

from argparse import ArgumentParser
from collections import OrderedDict

import ROOT
from array import array

ROOT.ROOT.EnableImplicitMT()
ROOT.gROOT.SetBatch(True)

import utils_root_canvas as utils

# LEPTON_SELECTIONS = ['single_muon']
LEPTON_SELECTIONS = ['ee', 'mumu', 'emu']
SGN_MASSES = ['500_4', '750_4', '1000_4', '1250_4', '1500_4', '1750_4', '2000_4', '2500_4', '3000_4', '4000_4']
MASSES = ['500', '750', '1000', '1250', '1500', '1750', '2000', '2500', '3000', '4000']
WIDTHS = ['4']#, '10', '20', '50']
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
VARIABLES = [
    'hotvr_invariant_mass_leading_subleading',
    # 'dilepton_invariant_mass_leading_subleading',
    # 'cutflow'
    # 'nhotvr'
]
for jet_type in ['leading']:#, 'subleading']:#:, 'subleading']:
    VARIABLES.extend([
        # f'hotvr_scoreBDT_{jet_type}',
        # f'hotvr_mass_{jet_type}',
        # f'hotvr_pt_{jet_type}',
        # f'hotvr_eta_{jet_type}',
        # f'hotvr_phi_{jet_type}',
#         # f'hotvr_pt_vs_mass_{jet_type}',
#         f'hotvr_tau3_over_tau2_{jet_type}',
#         f'hotvr_fractional_subjet_pt_{jet_type}',
#         f'hotvr_scoreBDT_{jet_type}',
#         f'hotvr_min_pairwise_subjets_mass_{jet_type}',
#         f'hotvr_nsubjets_{jet_type}',
        # f'ak4_outside_hotvr_pt_{jet_type}', 
        # f'ak4_outside_hotvr_eta_{jet_type}', 
        # f'ak4_outside_hotvr_phi_{jet_type}', 
        # f'{jet_type}_pt',
        # f'{jet_type}_eta',
        # f'{jet_type}_phi',
        # f'{jet_type}_pfRelIso04_all'
    ])

REGION = 'SR1b2T'
# ERA = 'all_years_Run2'

if 'SR2b' in REGION:
    EVENT_SELECTION = 'after_2OS_off_Zpeak_2ak4_2b_outside_hotvr_2_hotvr'
if 'SR1b' in REGION:
    EVENT_SELECTION = 'after_2OS_off_Zpeak_2ak4_1b_outside_hotvr_2_hotvr'
if 'CR' in REGION:
    LEPTON_SELECTIONS = ['ee', 'mumu']
if 'single_muon' in REGION:
    LEPTON_SELECTIONS = ['single_muon']

FLAVORS = [
    'hadronic_t', 
    # 'pure_qcd', #_or_hadronic_t', 
    # 'b_quark',
    # 'one_b_quark',
    # 'one_b_quark_and_no_hadronic_t',
    # 'no_b_quark', 
    # 'no_b_quark_and_no_hadronic_t',
    # 'b_quark_or_hadronic_t',
]

is_tta = False
is_tth = False

JET_COMPOSITION = True

class Processor:
    def __init__(self, input_dir, output_dir, year):
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.year = year

        self.output_dir = self._creation_output_dirs()

    def _creation_output_dirs(self):
        # check if dir exist
        self.output_dirs = {}
        
        for variable in VARIABLES:
            sub_output_dir = ""
            if 'hotvr_variables' in self.input_dir:
                sub_output_dir = os.path.join(
                    self.output_dir, 
                    self.year, 
                    # f"{self.input_dir}_{REGION}_DY_rescaling/{variable}"
                    f"{self.input_dir}_{REGION}/{variable}"
                )
            else:
                sub_output_dir = os.path.join(
                    self.output_dir, 
                    self.year, 
                    f"{self.input_dir}_{REGION}/{variable}"
                )
            print(sub_output_dir)
            if not os.path.exists(sub_output_dir):
                os.makedirs(sub_output_dir)
            self.output_dirs[variable] = sub_output_dir
        
        return self.output_dirs

    def add_uncorr_syst(self, merged_file, files_per_year, years, systematic, variation, lepton_selection):
        # target dir e.g. triggerUp/lepton_selection

        # loop over all histograms once, then accumulate deltas across years
        nom_dir = merged_file.GetDirectory(f"nominal/{lepton_selection}")
        if not nom_dir:
            return

        for k in nom_dir.GetListOfKeys():
            h_nom = k.ReadObj()
            name = h_nom.GetName()
            
            if 'tZq' in name or 'data' in name: 
                continue

            # add deltas for each year
            for year in years:
                year_dir = f"{systematic}{variation}_{year}/{lepton_selection}"
                sys_dir = merged_file.GetDirectory(year_dir)
                if not sys_dir:
                    continue

                # initialize uncorrelated hist with nominal
                h_uncorr = merged_file.Get(f"{year_dir}/{name}")

                f_year = ROOT.TFile.Open(files_per_year[year], "READ")
                nom_y_dir = f_year.GetDirectory(f"nominal/{lepton_selection}")
                if not nom_y_dir:
                    f_year.Close()
                    continue

                h_sys_y = sys_dir.Get(name)
                h_nom_y = nom_y_dir.Get(name)
                if h_sys_y and h_nom_y:
                    h_delta = h_sys_y.Clone("tmp")
                    h_delta.Add(h_nom_y, -1.0)   # sys − nom for that year
                    h_uncorr.Reset()
                    h_uncorr.Add(h_delta)
                    h_uncorr.Add(h_nom)
                    merged_file.cd(year_dir)
                    h_uncorr.Write("", ROOT.TObject.kOverwrite)
                    h_delta.Delete()
                else:
                    print(f"Missing {name} in {year_dir}")
                f_year.Close()

        # after all years, write the final uncorrelated histogram once
        # out_dir.cd()
        # h_uncorr.Write("", ROOT.TObject.kOverwrite)
        # sys.exit()

    def process(self):
        years = ['2016', '2016preVFP', '2017', '2018'] if self.year == 'all_years_Run2' else ['2022', '2022EE']

        for variable in VARIABLES:
            if 'score' in variable and 'T' in REGION:
                continue

            files = {} # collect per variable
            for year in years:
                fpath = f"analysis_outputs/{year}/{self.input_dir}_{REGION}/{variable}/distributions.root"
                if is_tta:
                    fpath = fpath.replace(".root", "_tta.root")
                if is_tth:
                    fpath = fpath.replace(".root", "_tth.root")
                
                if os.path.exists(fpath):
                    print(f"Analyzing file: {fpath}")
                    files[year] = fpath
                else:
                    print(f"Missing: {fpath}")

            if not files:
                print(f"No inputs for variable {variable}")
                continue

            output_filename = f"{self.output_dirs[variable]}/distributions"
            if is_tta:
                output_filename += "_tta"
            if is_tth:
                output_filename += "_tth"

            os.makedirs(self.output_dirs[variable], exist_ok=True)
            out_path = f"{output_filename}.root"

            cmd = (
                ["hadd"]
                + (["-f"] if os.path.exists(out_path) else [])
                + [out_path]
                + list(files.values())
            )
            subprocess.run(cmd, check=True)

            # merging uncorrelated systematics correctly
            merged_file = ROOT.TFile.Open(out_path, "UPDATE")
            files_per_year = {year: files[year] for year in years}

            for systematic in utils.UNCORRELATED_SYSTEMATICS:
                for variation in ["Up", "Down"]:
                    for lepton_selection in LEPTON_SELECTIONS:
                        self.add_uncorr_syst(merged_file, files_per_year, years, systematic, variation, lepton_selection)
            
            merged_file.Close()
        
            print(f"Saving distributions in {out_path}")

def main(
        input_dir, output_dir, 
        year):

    print("Analyzing file: {}".format(input_dir))

    processor = Processor(input_dir, output_dir, year)
    processor.process()


#################################################

def parse_args(argv=None):
    parser = ArgumentParser()

    parser.add_argument('--input_dir', type=str,
        help="Input file, where to find the h5 files")
    parser.add_argument('--output_dir', type=str,
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
