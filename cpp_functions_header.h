#include "ROOT/RDataFrame.hxx"
#include "ROOT/RVec.hxx"
#include "TCanvas.h"
#include "TH1D.h"
#include "TMath.h"
#include "TLatex.h"
#include "TStyle.h"
#include "TFile.h"
#include "TH2D.h"

#include <iostream>
#include <vector>
 
using namespace ROOT;
using namespace ROOT::VecOps;
using RNode = ROOT::RDF::RNode;

bool hadronic_genTop_fromReso_filter(const RVec<int> &reso_flag, const RVec<int> &hadronic_flag)
{
    std::vector<int> genTop_fromReso_indices;

    for(int itop = 0; itop < reso_flag.size(); ++itop)
    {
        if(reso_flag[itop]) {
            genTop_fromReso_indices.push_back(itop);
        }
    }

    if( hadronic_flag[genTop_fromReso_indices[0]] && hadronic_flag[genTop_fromReso_indices[1]])
    {
        return true;
    }
    else{return false;}
}

RVec<float> var_pureQCD_jets(const RVec<float> &var, const RVec<int> &flag1, const RVec<int> &flag2, const RVec<int> &flag3, 
    const RVec<int> &flag4, const RVec<int> &flag5, const RVec<int> &flag6, const RVec<int> &flag7, const RVec<int> &flag8 )
{
    RVec<float> var_QCD;
    for(int ijet = 0; ijet < var.size(); ++ijet)
    {
        if( flag1[ijet] || flag2[ijet] || flag3[ijet] || flag4[ijet] || flag5[ijet] || flag6[ijet] || flag7[ijet] || flag8[ijet])
        {
            var_QCD.push_back(var[ijet]);
        }
    }

    return var_QCD;
}

int multiplicity_pureQCD_jets(const RVec<float> &var, const RVec<int> &flag1, const RVec<int> &flag2, const RVec<int> &flag3, 
    const RVec<int> &flag4, const RVec<int> &flag5, const RVec<int> &flag6, const RVec<int> &flag7, const RVec<int> &flag8 )
{
    int multiplicity = 0;
    for(int ijet = 0; ijet < var.size(); ++ijet)
    {
        if( flag1[ijet] || flag2[ijet] || flag3[ijet] || flag4[ijet] || flag5[ijet] || flag6[ijet] || flag7[ijet] || flag8[ijet])
        {
            multiplicity += 1;
        }
    }

    return multiplicity;
}

RVec<float> var_flavored_jets(const RVec<float> &var, const RVec<int> &flag)
{
    RVec<float> var_QCD;
    for(int ijet = 0; ijet < var.size(); ++ijet)
    {
        if(flag[ijet])
        {
            var_QCD.push_back(var[ijet]);
        }
    }

    return var_QCD;
}

int multiplicity_flavored_jets(const RVec<float> &var, const RVec<int> &flag)
{
    int multiplicity = 0;
    for(int ijet = 0; ijet < var.size(); ++ijet)
    {
        if(flag[ijet])
        {
            multiplicity += 1;
        }
    }

    return multiplicity;
}

float minimum_top_tagger_score(const RVec<float> &top_tagger_score)
// retrieving the minimum of the top tagger score (no matter the multiplicity)
{
    float min_top_tagger_score = 1.1; 
    for(size_t ijet=0; ijet < top_tagger_score.size(); ++ijet)
    {
        min_top_tagger_score = std::min(min_top_tagger_score, top_tagger_score[ijet] );
    }
   return min_top_tagger_score;
}

float invariant_mass(const RVec<float> &object_pt, const RVec<float> &object_eta, const RVec<float> &object_phi, const RVec<float> &object_mass)
// this function assumes at least two objects in the event;
// make sure is the case in the analysis modules (e.g. using ROOTDataFrame.Filter(...))
{
    float inv_mass = -1;

    if(object_pt.size() > 1)
    {
        TLorentzVector leading_object;
        TLorentzVector subleading_object;
        TLorentzVector objects_system;

        leading_object.SetPtEtaPhiM(object_pt[0], object_eta[0], object_phi[0], object_mass[0]);
        subleading_object.SetPtEtaPhiM(object_pt[1], object_eta[1], object_phi[1], object_mass[1]);

        objects_system = leading_object + subleading_object;
        inv_mass = objects_system.M();
    }

    return inv_mass;
}

float invariant_mass_1_gen_matched_1_gen_unmatched(
    const RVec<float> &gen_matched_object_pt, const RVec<float> &gen_matched_object_eta, const RVec<float> &gen_matched_object_phi, const RVec<float> &gen_matched_object_mass,
    const RVec<float> &gen_unmatched_object_pt, const RVec<float> &gen_unmatched_object_eta, const RVec<float> &gen_unmatched_object_phi, const RVec<float> &gen_unmatched_object_mass)
// this function assumes exactly 1 gen_matched + 1 gen_unmatched;
// make sure is the case in the analysis modules (e.g. using ROOTDataFrame.Filter(...))
{
    TLorentzVector gen_matched;
    TLorentzVector gen_unmatched;
    TLorentzVector objects_system;

    gen_matched.SetPtEtaPhiM(gen_matched_object_pt[0], gen_matched_object_eta[0], gen_matched_object_phi[0], gen_matched_object_mass[0]);
    gen_unmatched.SetPtEtaPhiM(gen_unmatched_object_pt[0], gen_unmatched_object_eta[0], gen_unmatched_object_phi[0], gen_unmatched_object_mass[0]);

    objects_system = gen_matched + gen_unmatched;

    return objects_system.M();

}