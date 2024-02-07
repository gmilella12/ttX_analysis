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

//////////////////////
//////////////////////

struct SubHOTVRJet
{
    int index;
    float pt;
    float eta;
    float phi;
    float mass;

    SubHOTVRJet(const int &subhotvr_index, const float &subhotvr_pt, const float &subhotvr_eta, const float &subhotvr_phi, const float &subhotvr_mass) : index(subhotvr_index), pt(subhotvr_pt), eta(subhotvr_eta), phi(subhotvr_phi), mass(subhotvr_mass) {}
};

struct pt_subjet_sorting
{
    inline bool operator() (const SubHOTVRJet& subjet1, const SubHOTVRJet& subjet2)
    {
        return (subjet1.pt > subjet2.pt);
    }   
};

Float_t deltaPhi(Float_t phi1, Float_t phi2)
{
    Float_t res;
    res = phi1 - phi2;
    while (  res > TMath::Pi())
    {
        res -= 2. * TMath::Pi();
    }
    while ( res <= -(TMath::Pi()))
    {
        res += 2. * TMath::Pi();
    }
    return res;
}

Float_t deltaR(Float_t delta_phi, Float_t eta1, Float_t eta2)
{
    return TMath::Sqrt( TMath::Power(delta_phi,2) + TMath::Power(eta1-eta2,2) );
}

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

float invariant_mass_emu(
    const RVec<float> &muon_pt, const RVec<float> &muon_eta, const RVec<float> &muon_phi, const RVec<float> &muon_mass,
    const RVec<float> &electron_pt, const RVec<float> &electron_eta, const RVec<float> &electron_phi, const RVec<float> &electron_mass)
// this function assumes exactly 1 muon + 1 electron;
// make sure is the case in the analysis modules (e.g. using ROOTDataFrame.Filter(...))
{
    float inv_mass = -1;

    if(muon_pt.size() == 1 & electron_pt.size() == 1)
    {
        TLorentzVector muon;
        TLorentzVector electron;
        TLorentzVector lepton_system;

        muon.SetPtEtaPhiM(muon_pt[0], muon_eta[0], muon_phi[0], muon_mass[0]);
        electron.SetPtEtaPhiM(electron_pt[0], electron_eta[0], electron_phi[0], electron_mass[0]);

        lepton_system = muon + electron;
        inv_mass = lepton_system.M();
    }

    return inv_mass;

}

RVec<float> tauY_over_tauX(const RVec<float> &tauY, const RVec<float> &tauX)
{
    RVec<float> tauY_div_tauX;
    for(int ijet = 0; ijet < tauY.size(); ++ijet)
    {
        tauY_div_tauX.push_back(tauY[ijet] / tauX[ijet]);
    }

    return tauY_div_tauX;
}

RVec<float> min_deltaR_X_Y(const RVec<float> &x_eta, const RVec<float> &x_phi, const RVec<float> &y_eta, const RVec<float> &y_phi)
{
    std::vector<float> min_delta_r_vec;
    for(size_t ix = 0; ix < x_eta.size(); ++ix)
        {
            float min_deltaR = 10.;
            if(y_eta.size() == 0)
            {
                min_delta_r_vec.push_back(-1); // needed otherwise it will try to enter a RVec with 0 items
            }
            for(size_t iy=0; iy < y_eta.size(); ++iy)
            {  
                min_deltaR = std::min(min_deltaR, deltaR(deltaPhi(y_phi[iy], x_phi[ix] ), y_eta[iy] , x_eta[ix] ) );
            }
            min_delta_r_vec.push_back(min_deltaR);
        }
   return min_delta_r_vec;
}

float delta_phi(const RVec<float> &phi)
{
    float deltaphi = -99;
    if(phi.size() >= 2)
        {
            deltaphi = deltaPhi(phi[0], phi[1]);
        }
    return deltaphi;
}

float HT(const RVec<float> &boosted_jet_pt, const RVec<float> &jet_pt)
{
    float ht;
    for(size_t iboostjet = 0; iboostjet < boosted_jet_pt.size(); ++iboostjet)
        { 
            ht += boosted_jet_pt[iboostjet];
        }
    for(size_t ijet = 0; ijet < jet_pt.size(); ++ijet)
        {
            ht += jet_pt[ijet];
        }
    return ht;
}

float HT_only_ak4(const RVec<float> &jet_pt)
{
    float ht;
    for(size_t ijet = 0; ijet < jet_pt.size(); ++ijet)
        {
            ht += jet_pt[ijet];
        }
    return ht;
}

float findMassMinimum(const std::vector<float>& subject_combinations_masses) {
    float min_mass = std::numeric_limits<int>::max();
    // Iterate over each element in the vector
    for (int mass : subject_combinations_masses) {
        if (mass < min_mass) {
            min_mass = mass;
        }
    }
    return min_mass;
}

RVec<int> hotvr_cut_based_top_tagged(
    const RVec<int> &hotvrs_subjet1_idx, const RVec<int> &hotvrs_subjet2_idx, const RVec<int> &hotvrs_subjet3_idx, 
    const RVec<float> &hotvrs_pt, const RVec<float> &hotvrs_eta, const RVec<float> &hotvrs_phi, const RVec<float> &hotvrs_mass, 
    const RVec<float> &hotvrs_tau3, const RVec<float> &hotvrs_tau2, const RVec<int> &subhotvrs_idx,  
    const RVec<float> &subhotvrs_pt, const RVec<float> &subhotvrs_eta, const RVec<float> &subhotvrs_phi, const RVec<float> &subhotvrs_mass)
{
    // bool is_top_tagged = false;
    RVec<int> hotvr_top_tagged;
    for(size_t ihotvr=0; ihotvr<hotvrs_subjet1_idx.size(); ++ihotvr)
    {
        std::vector <SubHOTVRJet> subjets_in_hotvr;

        for(size_t isubhotvr=0; isubhotvr<subhotvrs_idx.size(); ++isubhotvr)
            {
                if(hotvrs_subjet1_idx[ihotvr]==subhotvrs_idx[isubhotvr] || hotvrs_subjet2_idx[ihotvr]==subhotvrs_idx[isubhotvr] || hotvrs_subjet3_idx[ihotvr]==subhotvrs_idx[isubhotvr])
                {
                    subjets_in_hotvr.push_back(SubHOTVRJet(subhotvrs_idx[isubhotvr], subhotvrs_pt[isubhotvr], subhotvrs_eta[isubhotvr], subhotvrs_phi[isubhotvr], subhotvrs_mass[isubhotvr]));
                }
                else{continue;}
            }

        // // pt, mass, tau3/tau2 cut on hotvr jet
        if( hotvrs_pt[ihotvr]>=200 && hotvrs_mass[ihotvr] > 140 && hotvrs_mass[ihotvr] < 220 && (hotvrs_tau3[ihotvr] / hotvrs_tau2[ihotvr]) < 0.56 && subjets_in_hotvr.size()>2)
        {   
            std::sort(subjets_in_hotvr.begin(), subjets_in_hotvr.end(), pt_subjet_sorting());
            TLorentzVector subjet1_lorentz;
            TLorentzVector subjet2_lorentz;
            TLorentzVector subjet3_lorentz;
        
            subjet1_lorentz.SetPtEtaPhiM(subjets_in_hotvr[0].pt,subjets_in_hotvr[0].eta,subjets_in_hotvr[0].phi,subjets_in_hotvr[0].mass);
            subjet2_lorentz.SetPtEtaPhiM(subjets_in_hotvr[1].pt,subjets_in_hotvr[1].eta,subjets_in_hotvr[1].phi,subjets_in_hotvr[1].mass);
            subjet3_lorentz.SetPtEtaPhiM(subjets_in_hotvr[2].pt,subjets_in_hotvr[2].eta,subjets_in_hotvr[2].phi,subjets_in_hotvr[2].mass);

            std::vector<float> subjet_combinations_masses;
            subjet_combinations_masses.push_back( (subjet1_lorentz + subjet2_lorentz).M() );
            subjet_combinations_masses.push_back( (subjet1_lorentz + subjet3_lorentz).M() );
            subjet_combinations_masses.push_back( (subjet2_lorentz + subjet3_lorentz).M() );

            float min_pairwise_mass = findMassMinimum(subjet_combinations_masses);

            // cut on min mass pairwise and relPt of the leading subjet
            if(min_pairwise_mass>50. && (subjets_in_hotvr[0].pt / hotvrs_pt[ihotvr]) < 0.8)
            {
                hotvr_top_tagged.push_back(1);
            }
            else{hotvr_top_tagged.push_back(0);}
        }
        else{hotvr_top_tagged.push_back(0);}
    }
    return hotvr_top_tagged;
}

RVec<float> min_deltaR_outside(
    const RVec<float> &jet_eta, const RVec<float> &jet_phi, const RVec<float> &boosted_jet_eta, const RVec<float> &boosted_jet_phi)
{
    std::vector<float> min_deltaR_vec; 
    for(size_t ijet=0; ijet < jet_eta.size(); ++ijet)
    {
            float min_deltaR = 10.;
            for(size_t iboosted=0; iboosted < boosted_jet_eta.size(); ++iboosted)
                {
                    min_deltaR = std::min(min_deltaR, deltaR(deltaPhi(boosted_jet_phi[iboosted], jet_phi[ijet] ), boosted_jet_eta[iboosted], jet_eta[ijet] ) );
                }
            min_deltaR_vec.push_back(min_deltaR);
    }
   return min_deltaR_vec;
}

RVec<float> max_deltaR_inside(
    const RVec<float> &jet_eta, const RVec<float> &jet_phi, const RVec<float> &boosted_jet_eta, const RVec<float> &boosted_jet_phi
)
{
    std::vector<float> max_deltaR_vec; 
    for(size_t ijet=0; ijet < jet_eta.size(); ++ijet)
    {
            float max_deltaR = 0.;
            for(size_t iboosted=0; iboosted < boosted_jet_eta.size(); ++iboosted)
                {
                    max_deltaR = std::max(max_deltaR, deltaR(deltaPhi(boosted_jet_phi[iboosted], jet_phi[ijet] ), boosted_jet_eta[iboosted], jet_eta[ijet] ) );
                }
            max_deltaR_vec.push_back(max_deltaR);
    }
   return max_deltaR_vec;
}