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

## Processing and merging files
After the files have been analyzed with the scripts, they can be merged PER YEAR for the different input files using `hadd MERGED_FILENAME.root ALL_PRODUCED_FILES*.root`. Once a merged file is created, it needs to be firstly processed in order to correctly format the processes and separate the uncertainties per year. Afterwards, processed files for different years need to be merged together. 

The scripts used for these steps are: 
```
python histo_processing.py --INPUT_DIR --year YEAR --OUTPUT_DIR
python histo_processing_theory_unc.py --INPUT_DIR --year YEAR --OUTPUT_DIR
```
The first script deals with the merging of similar MC process and the renaming of the templates. The second step processes the theoretical uncertainties creating a copy of the output file from the first step with the correct handling of them. One can specify the INPUT_DIR where the merged root file is located. In the script, one can specify the region and the variable under investigation. 

```
python histo_adding_all_years.py --INPUT_DIR --year YEAR --OUTPUT_DIR
```
which adds the processed files together for full Run2/3, processing correctly the uncorrelated or correlated systematics. The YEAR option can be `all_years_Run2` or `all_years_Run3`. One can specify the input directory (the naming should be similar for different years!), as well as the region and the variable under investigation.