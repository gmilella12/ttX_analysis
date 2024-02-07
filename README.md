# ttX_analysis
Useful scripts to analyze the ntuples produced with nano-AOD-tools (https://github.com/ttXcubed/nanoAOD-tools)
The ntuples can be found in ```/nfs/dust/cms/user/gmilella/ttX_ntuplizer```, separated per bkg, sgn, data samples and per years. 
The analysis scripts uses ROOT DataFrame (https://root.cern/doc/master/classROOT_1_1RDataFrame.html). In this repository, some examples to study AK8 variables are given. 
The scripts works per single file and includes:
* parsing of input file (importing x-section and MC genweight depending on the processes)
* creation of the output file
* creation of additional columns (not present in the file `TTree`) and event filtering
* histogram making which are then saved in the output file

## Running the scripts
A simple test on a single file can be done using the following command: 
```
python ak8_multiplicity.py --input_file FILE --output_dir OUTPUT_DIR --year YEAR
```
where FILE can be fetched from the aforementioned repository (e.g. `/nfs/dust/cms/user/gmilella/ttX_ntuplizer/bkg_2018_hotvr/merged/tt_dilepton_MC2018_ntuplizer_5_merged.root`). OUTPUT_DIR and YEAR are choosen by the user.
Additional option that can be used are:
* `is_sgn` or `is_data`
* `sys`: specifying the systematic (and the up/down variations) for which doing the analysis. 
The various systematics can be read from the ntuples

*N.B.*: 
* the script fetches the name of the process from the file (e.g `tt_dilepton`) which is then used to fetch its cross-section in the `xsec.yaml`.
Adjust the file name accordingly!
* in case of bkg files, I have split the same processes in multiple files to make the analysis faster. Therefore in the repository where the ntuples are fetched, there is an additional file `sum_gen_weights.yaml` that sums up the genweights for all the split files per process. 
* the scripts (current version: `commit 54c73fd`) select event with exactly 2 opposite sign leptons, at least 2 AK4 (2b) to be outside the HOTVR jet cone (if the latter is present in an event)

## Condor submission
Parallel analysis of files can be done by sendind as many condor jobs as ntuples. This can be done with the following commands:
```
python ttX_analysis_submission_template.py --OUTPUT_DIR --year YEAR
condor_submit ttX_analysis_condor_submission_new
```
additional arguments are `is_sgn` or `is_data`. 
The first command generates the `ttX_analysis_condor_submission_new` and the executable file depending on the arguments given. 
The ntuples are fetched automatically in the `ttX_analysis_submission_template.py`. You would need to modify the path according to where your ntuples are stored

