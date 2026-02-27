#ifndef CPP_MC_SCALING_HEADER_H
#define CPP_MC_SCALING_HEADER_H

#include <TFile.h>
#include <TH1F.h>
#include <iostream>
#include <ROOT/RVec.hxx>
#include <string>

namespace get_mc_scaling {

class ScaleFactorHelper {
public:
    static ScaleFactorHelper* getInstance(const std::string& region, const std::string& lepton_selection) {
        return new ScaleFactorHelper(region, lepton_selection);
    }

    std::pair<float, float> getScaleFactorAndError(float inv_mass) {
        if (!histo) {
            std::cerr << "❌ Histogram not initialized!" << std::endl;
            return {1.0, 0.0};
        }

        int binX = histo->GetXaxis()->FindBin(inv_mass);
        float scale_factor = histo->GetBinContent(binX);
        float scale_error = histo->GetBinError(binX);
        return {scale_factor, scale_error}; 
    }

public:  // Change from private to protected so it is accessible outside the class
    TH1F* histo;

private:
    TFile* scale_file;

    ScaleFactorHelper(const std::string& region, const std::string& lepton_selection) {
        std::string file_path = getFilePath(region);
        scale_file = new TFile(file_path.c_str(), "READ");

        if (!scale_file || !scale_file->IsOpen()) {
            // std::cerr << "Failed to open file: " << file_path << std::endl;
            histo = nullptr;
            return;
        } 

        // std::cout << "MC scaling file opened: " << file_path << std::endl;
        
        std::string histo_path = "nominal/" + lepton_selection + "/data_over_mc_" + lepton_selection + "_hotvr_invariant_mass_leading_subleading";
        histo = dynamic_cast<TH1F*>(scale_file->Get(histo_path.c_str()));

        if (!histo) {
            // std::cerr << "Error: Histogram " << histo_path << " not found in file!" << std::endl;
            scale_file->Close();
            histo = nullptr;
        } else {
            // std::cout << "Histogram found: " << histo_path << std::endl;
        }
    }

    // Correctly define getFilePath()
    std::string getFilePath(const std::string& region) {
        if (region == "SR1b0T") {
            return "mc_scaling_for_bkg_estimation/SR1b0T/hotvr_invariant_mass_leading_subleading.root";
        } 
        else if (region == "SR2b0T") {
            return "mc_scaling_for_bkg_estimation/SR2b0T/hotvr_invariant_mass_leading_subleading.root";
        } 
        std::cerr << "Error: Region " << region << " is not recognized!" << std::endl;
        std::exit(EXIT_FAILURE);
    }
};

// Fully define `rescaling_weights()` in the header using `inline`
inline std::pair<float, float> rescaling_weights(
    const float object_inv_mass, 
    const std::string& region, 
    const std::string& lepton_selection) 
{
    if (object_inv_mass == -1.0) {  
        return {1.0, 0.0};
    }

    ScaleFactorHelper* helper = ScaleFactorHelper::getInstance(region, lepton_selection);
    
    // 🔹 Check if `histo` exists before using it
    if (!helper || !(helper->histo)) {
        // std::cerr << "Error: Histogram not found! Returning default scale factor." << std::endl;
        delete helper;  // Free memory
        return {1.0, 0.0};
    }

    std::pair<float, float> result = helper->getScaleFactorAndError(object_inv_mass);
    delete helper;  // Free memory after use

    return result;
}

} // namespace get_mc_scaling

#endif // CPP_MC_SCALING_HEADER_H
