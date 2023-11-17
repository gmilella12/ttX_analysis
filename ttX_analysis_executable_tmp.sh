#!/bin/bash
source /cvmfs/grid.desy.de/etc/profile.d/grid-ui-env.sh
source /cvmfs/cms.cern.ch/cmsset_default.sh
export VO_CMS_SW_DIR=/cvmfs/cms.cern.ch
source $VO_CMS_SW_DIR/cmsset_default.sh
export CMSSW_GIT_REFERENCE=/cvmfs/cms.cern.ch/cmssw.git.daily
#source $VO_CMS_SW_DIR/crab3/crab.sh                                                                                                                                                                       
#source $VO_CMS_SW_DIR/cvmfs/cms.cern.ch/common/crab-setup.sh                                                                                                                                              
source /cvmfs/cms.cern.ch/cmsset_default.sh
source /cvmfs/cms.cern.ch/common/crab-setup.sh


cd /afs/desy.de/user/g/gmilella/ttx3_analysis/CMSSW_11_1_7/src
eval `scramv1 runtime -sh`

cd /afs/desy.de/user/g/gmilella/ttX3_post_ntuplization_analysis/

echo "ARGS: $@"

#cd $_CONDOR_SCRATCH_DIR

hostname
date
pwd
python ANALYZER.py $@ --year YEAR
date


