#include "ROOT/RDataFrame.hxx"
#include "ROOT/RVec.hxx"
#include "TCanvas.h"
#include "TH1.h"
#include "TMath.h"
#include "TLatex.h"
#include "TStyle.h"
#include "TFile.h"
#include "TH2D.h"

#include <cmath>

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

float invariant_mass_1tag_category(
    const RVec<float> &tagged_pt, 
    const RVec<float> &tagged_eta, 
    const RVec<float> &tagged_phi, 
    const RVec<float> &tagged_mass,
    const RVec<float> &tagged_index,
    const RVec<float> &jet_pt, 
    const RVec<float> &jet_eta, 
    const RVec<float> &jet_phi, 
    const RVec<float> &jet_mass,
    const RVec<float> &jet_index
)
{
    float inv_mass = -1;

    if(tagged_pt.size() == 1)
    {
        for(size_t ijet = 0; ijet < jet_pt.size(); ++ijet)
        {
            if(jet_index[ijet] == tagged_index[0]){continue;}
            else
            {
                TLorentzVector tagged_object;
                TLorentzVector jet_object;
                TLorentzVector objects_system;

                tagged_object.SetPtEtaPhiM(tagged_pt[0], tagged_eta[0], tagged_phi[0], tagged_mass[0]);
                jet_object.SetPtEtaPhiM(jet_pt[ijet], jet_eta[ijet], jet_phi[ijet], jet_mass[ijet]);

                objects_system = tagged_object + jet_object;
                inv_mass = objects_system.M();

                return inv_mass;
            }
        } 
    }

    // Explicitly return inv_mass for cases where no value was returned in the loop
    return inv_mass;
}


float getScaleFactor(float pt) {
    TFile eff_file("tagging_hotvr_efficiency_vs_pt.root");
    if (!eff_file.IsOpen()) {
        std::cerr << "Failed to open file!" << std::endl;
        return 1.0;
    }

    TH1D* histo = dynamic_cast<TH1D*>(eff_file.Get("hotvr_tagging_efficiency"));
    if (!histo) {
        std::cerr << "Histogram not found!" << std::endl;
        eff_file.Close();
        return 1.0;
    }

    int bin = histo->FindBin(pt);
    float scaleFactor = histo->GetBinContent(bin);

    eff_file.Close();
    return scaleFactor;
}

float rescaling_weights(const RVec<float> &object_pt) {
    float scale_jet1 = getScaleFactor(object_pt[0]);
    float scale_jet2 = getScaleFactor(object_pt[1]);

    return scale_jet1 * scale_jet2;
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
    float ht = -99;
    if(boosted_jet_pt.size() == 0){return ht;}
    else
    {
        ht = 0;
        for(size_t ijet = 0; ijet < jet_pt.size(); ++ijet)
        {
            ht += jet_pt[ijet];
        }
    for(size_t iboostjet = 0; iboostjet < boosted_jet_pt.size(); ++iboostjet)
        { 
            ht += boosted_jet_pt[iboostjet];
        }
    return ht;
    }
}

float HT_only_ak4(const RVec<float> &jet_pt)
{
    float ht = -99;
    if(jet_pt.size() > 0){ht = 0;}
    for(size_t ijet = 0; ijet < jet_pt.size(); ++ijet)
        {
            ht += jet_pt[ijet];
        }
    return ht;
}

float findMassMinimum(const std::vector<float>& subject_combinations_masses) {
    float min_mass = std::numeric_limits<float>::max();
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

// GEN MATCHING - LEPTON OBJECT
RVec<float> deltaR_reco_gen(
    const RVec<float> &reco_eta, const RVec<float> &reco_phi, const RVec<float> &reco_genPartIdx,
    const RVec<float> &gen_eta, const RVec<float> &gen_phi, const RVec<float> &gen_index
)
{
    std::vector<float> deltaR_vec; 
    for(size_t ireco=0; ireco < reco_eta.size(); ++ireco)
    {
        for(size_t igen=0; igen < gen_eta.size(); ++igen)
        {
            if( reco_genPartIdx[ireco] == gen_index[igen])
            {
                deltaR_vec.push_back(deltaR(deltaPhi(reco_phi[ireco], gen_phi[igen]), reco_eta[ireco], gen_eta[igen] ));
            }
            else{continue;}
        }
    }

    return deltaR_vec;
} 

RVec<float> deltapT_reco_gen(
    const RVec<float> &reco_pt, const RVec<float> &reco_genPartIdx,
    const RVec<float> &gen_pt, const RVec<float> &gen_index
)
{
    std::vector<float> deltapT_vec; 
    for(size_t ireco=0; ireco < reco_pt.size(); ++ireco)
    {
        for(size_t igen=0; igen < gen_pt.size(); ++igen)
        {
            if( reco_genPartIdx[ireco] == gen_index[igen])
            {
                deltapT_vec.push_back( ( gen_pt[igen] - reco_pt[ireco] ) / (0.5 * ( reco_pt[ireco] + gen_pt[igen] )) );
            }
            else{continue;}
        }
    }

    return deltapT_vec;
}

RVec<int> is_top_tagged(
    const RVec<float> &score, Float_t wp
)
{
    std::vector<int> jets_tagging_flags;
    for(size_t ijet = 0; ijet < score.size(); ++ijet) {
        if(score[ijet] >= wp) {
            jets_tagging_flags.push_back(1);
        } else {
            jets_tagging_flags.push_back(0);
        }
    }
    return jets_tagging_flags;
}

float deltaR_jets(
    const RVec<float> &jet_eta, const RVec<float> &jet_phi
)
{
    if (jet_eta.size() < 2) {
        return -1;
    }
    return deltaR(deltaPhi(jet_phi[0], jet_phi[1]), jet_eta[0], jet_eta[1] );
} 

int cut_and_count_categorization(
    const RVec<float> &score, Float_t wp, Float_t njet
)
{
    if (njet >= 2) {
        if (score[0] < wp && score[1] < wp) return 0; // 0th category: >=2 HOTVR no tagged
        else if (score[0] >= wp && score[1] < wp) return 1; // 1st category: >=2 HOTVR leading tagged, subleading not tagged
        else if (score[0] < wp && score[1] >= wp) return 2; // 2nd category: >=2 HOTVR leading not tagged, subleading tagged
        else if (score[0] >= wp && score[1] >= wp) return 3; // 3rd category: >=2 HOTVR both tagged
    } else if (njet == 1) {
        if (score[0] >= wp) return 4; // 4th category: ==1 HOTVR tagged
        else return 5; // 5th category: ==1 HOTVR not tagged
    }
    return -1;

}

int cut_and_count_categorization_cut_based(
    const RVec<float> &is_tagged, Float_t njet
)
{
    if (njet >= 2) {
        if (is_tagged[0] == 0 && is_tagged[1] == 0) return 0; // 0th category: >=2 HOTVR no tagged
        else if (is_tagged[0] && is_tagged[1] == 0) return 1; // 1st category: >=2 HOTVR leading tagged, subleading not tagged
        else if (is_tagged[0] == 0 && is_tagged[1] ) return 2; // 2nd category: >=2 HOTVR leading not tagged, subleading tagged
        else if (is_tagged[0] && is_tagged[1]) return 3; // 3rd category: >=2 HOTVR both tagged
    } else if (njet == 1) {
        if (is_tagged[0] == 0) return 4; // 4th category: ==1 HOTVR tagged
        else return 5; // 5th category: ==1 HOTVR not tagged
    }
    return -1;

}

// --- CUT FLOW FOR SIGNAL REGION
// int cut_flow_same_flavour_leptons(
//     Int_t trigger, Int_t nLeptons, const RVec<float> &leptonCharge, const RVec<float> &leptonPt, 
//     Float_t invariant_diLeptons_mass, Int_t nAK4Jets, const RVec<int> ak4IsInside, const RVec<int> ak4JetId, const RVec<float> ak4DeepFlavB,
//     Int_t nBJets, Int_t nBoostedJets, const RVec<float> hotvrBDT, const std::string &year
// ) {

//     std::map<std::string,float> b_tagging_wp = {{"2018", 0.0490}, {"2017", 0.0532}, {"2016", 0.0480}, {"2016preVFP", 0.0614} }; // loose
//     // std::map<std::string,float> b_tagging_wp = {{"2018", 0.2783}, {"2017", 0.3040}, {"2016", 0.2489}, {"2016preVFP", 0.3093} }; // medium
//     // std::map<std::string,float> b_tagging_wp = {{"2018", 0.7100}, {"2017", 0.7476}, {"2016", 0.6377}, {"2016preVFP", 0.7221} }; // tight
    
//     int cut_stage = 0;

//     // Trigger
//     if (!trigger) return cut_stage;
//     cut_stage = 1;

//     // ==2 leptons
//     if (nLeptons != 2) return cut_stage;
//     cut_stage += 1;

//     // OS + pT requirements + lowDileptonInvariantMass
//     if (!(leptonCharge[0] * leptonCharge[1] < 0 && leptonPt[0] > 25 && leptonPt[1] > 15 && invariant_diLeptons_mass > 20))
//         return cut_stage;
//     cut_stage += 1;

//     // Off Z peak
//     bool isOnZPeak = invariant_diLeptons_mass >= 80 && invariant_diLeptons_mass <= 101;
//     if (isOnZPeak) return cut_stage;
//     cut_stage += 1;

//     // at least 2 ak4
//     // if (nAK4Jets < 2) return cut_stage;
//     // if (nAK4Jets != 1) return cut_stage;
//     // cut_stage += 1;


//     // at least 2 ak4 with ak4IsInside == 0
//     int ak4OutsideCount = 0;
//     for (size_t i = 0; i < ak4IsInside.size(); ++i) {
//         if (ak4IsInside[i] == 0 && ak4JetId[i] & 0b10) ak4OutsideCount++;
//     }
//     if (ak4OutsideCount < 2) return cut_stage;
//     cut_stage += 1;
//     // if (ak4OutsideCount != 1) return cut_stage;
//     // cut_stage += 1;

//     // at least 2 ak4 with ak4DeepFlavB > 0.02
//     int ak4DeepFlavBCount = 0;
//     for (size_t i = 0; i < ak4DeepFlavB.size(); ++i) {
//         if (ak4DeepFlavB[i] > b_tagging_wp.at(year) && ak4IsInside[i] == 0 && ak4JetId[i] & 0b10) ak4DeepFlavBCount++;
//     }
//     // if (ak4DeepFlavBCount < 2) return cut_stage;
//     // cut_stage += 1;
//     if (ak4DeepFlavBCount != 1) return cut_stage; 
//     cut_stage += 1;

//     // at least hotvr
//     if (nBoostedJets < 2) return cut_stage;
//     cut_stage += 1;
    
//     // both tagged
//     if (hotvrBDT[0]<0.5 || hotvrBDT[1]<0.5) return cut_stage;
//     cut_stage += 1;
//     // at least one tagged
//     // if (hotvrBDT[0] < 0.5 && hotvrBDT[1] < 0.5) return cut_stage;
//     // cut_stage += 1;

//     return cut_stage;
// }
// ---

// --- CUT FLOW FOR CONTROL REGION
int cut_flow_same_flavour_leptons(
    Int_t trigger, Int_t nLeptons, const RVec<float> &leptonCharge, const RVec<float> &leptonPt, 
    Float_t invariant_diLeptons_mass, Int_t nAK4Jets, const RVec<int> ak4IsInside, const RVec<int> ak4JetId, const RVec<float> ak4DeepFlavB,
    Int_t nBJets, Int_t nBoostedJets, const RVec<float> hotvrBDT, const std::string &year
) {

    std::map<std::string,float> b_tagging_wp = {{"2018", 0.0490}, {"2017", 0.0532}, {"2016", 0.0480}, {"2016preVFP", 0.0614} };
    // 
    int cut_stage = 0;

    // Trigger
    if (!trigger) return cut_stage;
    cut_stage = 1;

    // ==2 leptons
    if (nLeptons != 2) return cut_stage;
    cut_stage += 1;

    // OS + pT requirements + lowDileptonInvariantMass
    if (!(leptonCharge[0] * leptonCharge[1] < 0 && leptonPt[0] > 25 && leptonPt[1] > 15 && invariant_diLeptons_mass > 20))
        return cut_stage;
    cut_stage += 1;

    // Off Z peak
    bool isOffZPeak = invariant_diLeptons_mass < 80 || invariant_diLeptons_mass > 101;
    if (isOffZPeak) return cut_stage;
    cut_stage += 1;

    // int ak4OutsideCount = 0;
    // for (size_t i = 0; i < ak4IsInside.size(); ++i) {
    //     if (ak4IsInside[i] == 0) ak4OutsideCount++;
    // }
    // if (ak4OutsideCount < 2) return cut_stage;
    // cut_stage += 1;
    // if (ak4OutsideCount != 1) return cut_stage;
    // cut_stage += 1;

    // at least hotvr
    if (nBoostedJets < 2) return cut_stage;
    cut_stage += 1;

    // both tagged
    // if (hotvrBDT[0]<0.5 || hotvrBDT[1]<0.5) return cut_stage;
    // cut_stage += 1;
    // at least one tagged
    if (hotvrBDT[0] < 0.5 && hotvrBDT[1] < 0.5) return cut_stage;
    cut_stage += 1;

    return cut_stage;
}
// ---

int cut_flow_opposite_flavour_leptons(
    Int_t trigger, Int_t nElectrons, Int_t nMuons,
    const RVec<float> &electronCharge, const RVec<float> &electronPt, 
    const RVec<float> &muonCharge, const RVec<float> &muonPt, 
    Float_t invariant_diLeptons_mass, Int_t nAK4Jets, const RVec<int> ak4IsInside, const RVec<int> ak4JetId, const RVec<float> ak4DeepFlavB,
    Int_t nBJets, Int_t nBoostedJets, const RVec<float> hotvrBDT, const std::string &year
) {

    std::map<std::string,float> b_tagging_wp = {{"2018", 0.0490}, {"2017", 0.0532}, {"2016", 0.0480}, {"2016preVFP", 0.0614} }; // loose
    // std::map<std::string,float> b_tagging_wp = {{"2018", 0.2783}, {"2017", 0.3040}, {"2016", 0.2489}, {"2016preVFP", 0.3093} }; // medium
    // std::map<std::string,float> b_tagging_wp = {{"2018", 0.7100}, {"2017", 0.7476}, {"2016", 0.6377}, {"2016preVFP", 0.7221} }; // tight
    // 
    int cut_stage = 0;

    // Trigger
    if (!trigger) return cut_stage;
    cut_stage = 1;

    // ==1 electron and ==1 muon
    if (!(nElectrons == 1 && nMuons == 1)) return cut_stage;
    cut_stage += 1;

    // OS + pT requirements + lowDileptonInvariantMass
    bool ptRequirementsAndOS = (electronCharge[0] * muonCharge[0] < 0) && 
                             ((electronPt[0] > 25 && muonPt[0] > 15) || 
                              (muonPt[0] > 25 && electronPt[0] > 15));  
    if (!ptRequirementsAndOS) return cut_stage;
    cut_stage += 1;

    // at least 2 ak4
    // if (nAK4Jets < 2) return cut_stage;
    // if (nAK4Jets != 1) return cut_stage;
    // cut_stage += 1;

    // at least 2 ak4 with ak4IsInside == 0
    int ak4OutsideCount = 0;
    for (size_t i = 0; i < ak4IsInside.size(); ++i) {
        if (ak4IsInside[i] == 0 && ak4JetId[i] & 0b10) ak4OutsideCount++;
    }
    if (ak4OutsideCount < 2) return cut_stage;
    cut_stage += 1;
    // if (ak4OutsideCount != 1) return cut_stage;
    // cut_stage += 1;

    // at least 2 ak4 with ak4DeepFlavB > 0.02
    int ak4DeepFlavBCount = 0;
    for (size_t i = 0; i < ak4DeepFlavB.size(); ++i) {
        if (ak4DeepFlavB[i] > b_tagging_wp.at(year) && ak4IsInside[i] == 0 && ak4JetId[i] & 0b10) ak4DeepFlavBCount++;
    }
    // if (ak4DeepFlavBCount < 2) return cut_stage;
    // cut_stage += 1;
    if (ak4DeepFlavBCount != 1) return cut_stage; 
    cut_stage += 1;

    // at least hotvr
    if (nBoostedJets < 2) return cut_stage;
    cut_stage += 1;

    // both tagged
    if (hotvrBDT[0]<0.5 || hotvrBDT[1]<0.5) return cut_stage;
    cut_stage += 1;
    // at least one tagged
    // if (hotvrBDT[0] < 0.5 && hotvrBDT[1] < 0.5) return cut_stage;
    // cut_stage += 1;

    return cut_stage;
}

float variable_per_jet_per_jet_composition(const RVec<float> &variable, const bool &composition_flag, int ijet)
{
    if(variable.size() >= 1)
    {
        if(composition_flag)
        {
            return variable[ijet];
        }
        else{return -99;}
    } 
    else{return -99;}
}

float variable_per_jet_per_jet_composition_non_covered(const RVec<float> &variable, const bool &composition_flag, int ijet)
{
    if(variable.size() >= 1)
    {
        if(composition_flag == false)
        {
            return variable[ijet];
        }
        else{return -99;}
    } 
    else{return -99;}
}

// int njet_per_jet_composition(const RVec<float> &variable, const bool &composition_flag)
// {
//     int njet = 0;
//     for(size_t ijet = 0; ijet < variable.size(); ++ijet)
//     {
//         if(composition_flag[ijet])
//         {
//             njet += 1;
//         }
//     }
//     return njet;
// }

// ELECTRON VARIABLES
bool dxy_dz_requirement(Float_t dxy, Float_t dz, Float_t eta)
{
    if ( TMath::Abs(eta) < 1.479)
    {
        return !(dxy>0.05 || dz>0.10);
    }
    else
    { 
        return !(dxy>0.10 || dz>0.20);
    }
}

bool dxy_dz_electron_cut(
    const RVec<float> &dxy, const RVec<float> &dz, const RVec<float> &eta
)
{
    if(dxy.size() == 2)
    {    
        return dxy_dz_requirement(dxy[0], dz[0], eta[0]) && dxy_dz_requirement(dxy[1], dz[1], eta[1]); 
    }
    else if(dxy.size() == 1)
    {
        return dxy_dz_requirement(dxy[0], dz[0], eta[0]);
    }
    else{return true;}
}

float deltaR_leptons(const RVec<float> &eta, const RVec<float> &phi)
{
   return deltaR(deltaPhi(phi[0], phi[1] ), eta[0], eta[1] );
}

// float me_ren_uncertainties()
// {
//     return ;
// }

// float me_norm_uncertainties()
// {
//     return ;
// }


// 

RVec<float> gentops_matched_to_tagged_jet_pt(
    const RVec<float> &gen_top_pt, const RVec<int> &gen_top_is_inside_hotvr_index, 
    const RVec<float> &hotvr_score, const RVec<int> &hotvr_index, Float_t score_wp
)
{
    std::vector<float> gen_top_pt_matched_to_tagged_jet; 
    for(size_t igen=0; igen < gen_top_pt.size(); ++igen)
    {
        int matched_idx = gen_top_is_inside_hotvr_index[igen];
        for(size_t ihotvr=0; ihotvr < hotvr_score.size(); ++ihotvr)
            if (hotvr_index[ihotvr] == matched_idx && hotvr_score[ihotvr]>score_wp)
            {
                gen_top_pt_matched_to_tagged_jet.push_back( gen_top_pt[igen] );
            }
    }

    return gen_top_pt_matched_to_tagged_jet;
}

int ngentops_matched_to_tagged_jet_pt(
    const RVec<float> &gen_top_pt, const RVec<int> &gen_top_is_inside_hotvr_index, 
    const RVec<float> &hotvr_score, const RVec<int> &hotvr_index, Float_t score_wp
)
{
    int ngen_top_pt_matched_to_tagged_jet;
    ngen_top_pt_matched_to_tagged_jet = 0;
    for(size_t igen=0; igen < gen_top_pt.size(); ++igen)
    {
        int matched_idx = gen_top_is_inside_hotvr_index[igen];
        for(size_t ihotvr=0; ihotvr < hotvr_score.size(); ++ihotvr)
            if (hotvr_index[ihotvr] == matched_idx && hotvr_score[ihotvr]>score_wp)
            {
                ngen_top_pt_matched_to_tagged_jet += 1;
            }
    }

    return ngen_top_pt_matched_to_tagged_jet;
}



bool contains(const RVec<int>& vec, int value) {
    for (const auto& element : vec) {
        if (element == value) {
            return true;
        }
    }
    return false;
}
RVec<float> matched_tagged_jet_to_gentop_var(
    const RVec<int> &gen_top_is_inside_hotvr_index, 
    const RVec<float> &hotvr_var, const RVec<float> &hotvr_score, const RVec<int> &hotvr_index, Float_t score_wp
)
{
    std::vector<float> tagged_jet_var_matched_to_top; 
    for(size_t ihotvr=0; ihotvr < hotvr_var.size(); ++ihotvr)
    {
        int matched_idx = hotvr_index[ihotvr];
        if (hotvr_score[ihotvr] >= score_wp && contains(gen_top_is_inside_hotvr_index, matched_idx))
        {
            tagged_jet_var_matched_to_top.push_back( hotvr_var[ihotvr] );
        }
    }

    return tagged_jet_var_matched_to_top;
}

RVec<float> matched_jet_to_gentop_var(
    const RVec<int> &gen_top_is_inside_hotvr_index, 
    const RVec<float> &hotvr_var, const RVec<int> &hotvr_index
)
{
    std::vector<float> jet_var_matched_to_top; 
    for(size_t ihotvr=0; ihotvr < hotvr_var.size(); ++ihotvr)
    {
        int matched_idx = hotvr_index[ihotvr];
        if (contains(gen_top_is_inside_hotvr_index, matched_idx))
        {
            jet_var_matched_to_top.push_back( hotvr_var[ihotvr] );
        }
    }
    return jet_var_matched_to_top;
}

bool HEM_veto(
    const RVec<float>& electron_pt,
    const RVec<float>& electron_eta,
    const RVec<float>& electron_phi, 
    const RVec<float>& jet_pt,
    const RVec<float>& jet_eta,
    const RVec<float>& jet_phi,
    const int run, 
    const bool isData
)
{
    const float HEM_eta_min = -3.2, HEM_eta_max = -1.3;
    const float HEM_phi_min_electron = -1.57, HEM_phi_max_electron = -0.87;
    const float HEM_phi_min_jet = -1.77, HEM_phi_max_jet = -0.67;
    const float pt_threshold = 20.0;

    if (isData && run < 319077) {
        return true;
    }

    for (size_t ie=0; ie < electron_pt.size(); ++ie) {
        if (electron_pt[ie] > pt_threshold &&
            electron_eta[ie] > HEM_eta_min && electron_eta[ie] < HEM_eta_max &&
            electron_phi[ie] > HEM_phi_min_electron && electron_phi[ie] < HEM_phi_max_electron) {
            return false; // Event fails HEM veto
        }
    }

    for (size_t ij=0; ij < electron_pt.size(); ++ij) {
        if (jet_pt[ij] > pt_threshold &&
            jet_eta[ij] > HEM_eta_min && jet_eta[ij] < HEM_eta_max &&
            jet_phi[ij] > HEM_phi_min_jet && jet_phi[ij] < HEM_phi_max_jet) {
            return false; // Event fails HEM veto
        }
    }

    return true;
}

RVec<int> is_inside_hotvr(
    const RVec<float> &jets_eta, 
    const RVec<float> &jets_phi, 
    const RVec<float> &hotvrs_pt, 
    const RVec<float> &hotvrs_eta, 
    const RVec<float> &hotvrs_phi, 
    const int nhotvr // this is the number of hotvrs that are expected to be cleaned against ak4 in an event 
)
{
    RVec<int> is_inside_vector(jets_eta.size(), 0);  
    if (hotvrs_eta.empty() || nhotvr == 0) return is_inside_vector;

    size_t hotvr_size = std::min(static_cast<size_t>(nhotvr), hotvrs_eta.size());

    for (size_t i = 0; i < jets_eta.size(); ++i) {
        float jet_eta = jets_eta[i];
        float jet_phi = jets_phi[i];

        for (size_t j = 0; j < hotvr_size; ++j) {
            float hotvr_eta = hotvrs_eta[j];
            float hotvr_phi = hotvrs_phi[j];
            float hotvr_pt = hotvrs_pt[j];

            float effective_radius = std::min(600.0f / hotvr_pt, 1.5f);

            float dEta = jet_eta - hotvr_eta;
            float dPhi = std::fabs(jet_phi - hotvr_phi);
            if (dPhi > M_PI) dPhi = 2 * M_PI - dPhi;

            float deltaR = std::sqrt(dEta * dEta + dPhi * dPhi);

            // If inside at least one HOTVR jet, set the flag and break
            if (deltaR < effective_radius) {
                is_inside_vector[i] = 1;
                break;
            }
        }
    }

    return is_inside_vector;
}