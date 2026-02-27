#ifndef CPP_TRIGGER_SCALE_FACTOR_HEADER_H
#define CPP_TRIGGER_SCALE_FACTOR_HEADER_H

#include <TFile.h>
#include <TH1D.h>
#include <TH2.h> // Correct type for histogram
#include <iostream>
#include <ROOT/RVec.hxx> // Include the RVec header
#include <string>        // Include string library

using namespace ROOT::VecOps; // Use the RVec namespace

namespace get_trigger_scale_factor {

class ScaleFactorHelper {
public:
    // Static method to get the instance of the class (Singleton pattern)
    static ScaleFactorHelper& getInstance(const std::string& year, const std::string& histo_name) {
        static ScaleFactorHelper instance(year, histo_name);
        return instance;
    }

    std::pair<float, float> getScaleFactorAndError(
            float pt1, 
            float pt2
        ) { // for method 3
        if (!histo) {
            std::cerr << "Histogram not initialized!" << std::endl;
            return {1.0, 0.0};
        }

        int binX = histo->GetXaxis()->FindBin(pt1); // pt1 = leading
        int binY = histo->GetYaxis()->FindBin(pt2); // pt2 = subleading

        float scale_factor = histo->GetBinContent(binX, binY);
        float scale_error = histo->GetBinError(binX, binY);

        // If either SF or error is NaN or negative, fallback
        if (!std::isfinite(scale_factor) || scale_factor <= 0.0f || !std::isfinite(scale_error) || scale_error < 0.0f) {
            // std::cerr << "Invalid scale factor or error (pt1=" << pt1 << ", pt2=" << pt2 
            //         << ") -> SF=" << scale_factor << ", Err=" << scale_error << std::endl;
            return {1.0f, 0.0f};
        }

        return {scale_factor, scale_error};

}

private:
    TFile* eff_file;
    TH2* histo;

    // Private constructor to prevent instantiation
    ScaleFactorHelper(const std::string& year, const std::string& histo_name) {
        std::string file_path = getFilePath(year);
        eff_file = new TFile(file_path.c_str());
        if (!eff_file->IsOpen()) {
            std::cerr << "Failed to open file: " << file_path << std::endl;
            histo = nullptr;
            return;
        } else {
            std::cout << "Tagging efficiency file opened: " << file_path << std::endl;
        }

        // Choose the appropriate histogram based on the is_data flag
        histo = dynamic_cast<TH2*>(eff_file->Get(histo_name.c_str())); // for data
        if (histo) {
            std::cout << "Tagging efficiency histogram loaded: " << histo_name << std::endl;
        }

        if (!histo) {
            std::cerr << "Histogram " << histo_name << " not found in file: " << file_path << std::endl;
            eff_file->Close();
            histo = nullptr;
            return;
        }
    }

    // Helper function to determine the correct file path based on the year
    std::string getFilePath(const std::string& year) {
        if (year == "2022") {
            return "trigger_efficiency/2022/DileptonTriggerSFs_2022pre.root";
        } 
        else if (year == "2022EE") {
            return "trigger_efficiency/2022EE/DileptonTriggerSFs_2022post.root";
        }
        else {
            std::cerr << "Year " << year << " is not recognized!" << std::endl;
            std::exit(EXIT_FAILURE);
        }
    }

    // Delete copy constructor and assignment operator
    ScaleFactorHelper(const ScaleFactorHelper&) = delete;
    ScaleFactorHelper& operator=(const ScaleFactorHelper&) = delete;

    ~ScaleFactorHelper() {
        if (eff_file) {
            eff_file->Close();
            delete eff_file;
        }
    }
};

// Function to calculate the rescaling weights
std::pair<float, float> rescaling_weights(
    const RVec<float>& object_pt, 
    const std::string& year, 
    const std::string& histo_name
    ) {
    if (object_pt.size() < 2) {
        return {1.0, 0.0};
    }

    ScaleFactorHelper& helper = ScaleFactorHelper::getInstance(year, histo_name);
    return helper.getScaleFactorAndError(object_pt[0], object_pt[1]); 
}

} // namespace get_trigger_scale_factor

#endif // CPP_TRIGGER_SCALE_FACTOR_HEADER_H
