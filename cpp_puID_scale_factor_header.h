#ifndef CPP_PUID_SCALE_FACTOR_HEADER_H
#define CPP_PUID_SCALE_FACTOR_HEADER_H

#include <TFile.h>
#include <TH1D.h>
#include <TH2F.h> // Correct type for histogram
#include <iostream>
#include <ROOT/RVec.hxx> // Include the RVec header
#include <string>        // Include string library

using namespace ROOT::VecOps; // Use the RVec namespace

namespace get_PUID_scale_factor {

class ScaleFactorHelper {
public:
    static ScaleFactorHelper& getInstance(const std::string& eff_name, const std::string& sf_name) {
        static ScaleFactorHelper instance(eff_name, sf_name);
        return instance;
    }

    std::pair<float, float> getScaleFactorAndError(
            const RVec<int>& jets_puID,
            const RVec<float>& jets_pt,
            const RVec<float>& jets_eta
        ) 
    { // for method 3
        if (!histo_eff) {
            std::cerr << "Histogram PUID eff map not initialized!" << std::endl;
            return {1.0, 0.0};
        }
        if (!histo_sf) {
            std::cerr << "Histogram PUID SF not initialized!" << std::endl;
            return {1.0, 0.0};
        }

        // based on method 1a https://twiki.cern.ch/twiki/bin/view/CMS/BTagSFMethods#1a_Event_reweighting_using_scale
        float weight = 1.0;
        float weight_error2 = 0.0;

        for (int ijet = 0; ijet < jets_pt.size(); ++ijet) {
            if (jets_pt[ijet] > 50){
                continue;
            }
            int binY = histo_sf->GetXaxis()->FindBin(jets_pt[ijet]); 
            int binX = histo_sf->GetYaxis()->FindBin(jets_eta[ijet]);

            float sf = histo_sf->GetBinContent(binX, binY);
            float sf_err = histo_sf->GetBinError(binX, binY);
            float eff = histo_eff->GetBinContent(binX, binY);
            float eff_err = histo_eff->GetBinError(binX, binY);

            if (sf <= 0 || eff <= 0) {
                continue;  // Or: sf = 1.0; eff = 1.0; err = 0.0;
            }

            float factor, err;

            if (jets_puID[ijet] != 0) {  // loose puID identified jet
                factor = sf * eff;
                err = factor * std::sqrt(std::pow(sf_err / sf, 2) + std::pow(eff_err / eff, 2));
            } else {  // not tagged
                factor = 1.0f - sf * eff;
                err = std::sqrt(std::pow(sf * eff_err, 2) + std::pow(eff * sf_err, 2));
            }

            weight *= factor;
            if (factor > 0)
                weight_error2 += std::pow(err / factor, 2);  // relative errors add in quadrature
        }

        float weight_error = weight * std::sqrt(weight_error2);
        return {weight, weight_error};
    }

private:
    TFile* eff_file;
    TH2F* histo_eff;
    TH2F* histo_sf;

    // Private constructor to prevent instantiation
    ScaleFactorHelper(const std::string& histo_eff_name, const std::string& histo_sf_name) {
        std::string file_path = getFilePath();
        eff_file = new TFile(file_path.c_str());
        if (!eff_file->IsOpen()) {
            std::cerr << "Failed to open file: " << file_path << std::endl;
            histo_eff = nullptr;
            histo_sf = nullptr;
            return;
        } else {
            std::cout << "PUID efficiency/SF file opened: " << file_path << std::endl;
        }

        // Choose the appropriate histogram based on the is_data flag
        histo_eff = dynamic_cast<TH2F*>(eff_file->Get(histo_eff_name.c_str()));
        histo_sf = dynamic_cast<TH2F*>(eff_file->Get(histo_sf_name.c_str()));
        if (!histo_eff || !histo_sf) {
            std::cerr << "Failed to load histograms: " << histo_eff_name << " or " << histo_sf_name << std::endl;
            eff_file->Close();
            histo_eff = nullptr;
            histo_sf = nullptr;
        } else {
            std::cout << "Loaded histograms " << histo_eff_name << " or " << histo_sf_name << "for PUID SF and efficiency." << std::endl;
        }
    }

    // Helper function to determine the correct file path based on the year
    std::string getFilePath() {
        return "PUID_efficiency_and_SFs/PUID_106XTraining_ULRun2_EffSFandUncties_v1.root";
    }

    ~ScaleFactorHelper() {
        if (eff_file) {
            eff_file->Close();
            delete eff_file;
        }
    }

    // Delete copy constructor and assignment operator
    ScaleFactorHelper(const ScaleFactorHelper&) = delete;
    ScaleFactorHelper& operator=(const ScaleFactorHelper&) = delete;
};

// Function to calculate the rescaling weights
std::pair<float, float> rescaling_weights(
    const RVec<int>& object_puID,
    const RVec<float>& object_pt,
    const RVec<float>& object_eta,
    const std::string& histo_eff_name,
    const std::string& histo_sf_name
    ) {
    if (object_pt.size() < 2) {
        return {1.0, 0.0};
    }

    ScaleFactorHelper& helper = ScaleFactorHelper::getInstance(histo_eff_name, histo_sf_name);
    return helper.getScaleFactorAndError(object_puID, object_pt, object_eta); 
}

} // namespace get_PUID_scale_factor

#endif // CPP_PUID_SCALE_FACTOR_HEADER_H
