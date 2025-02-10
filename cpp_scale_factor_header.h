#ifndef CPP_FUNCTIONS_HEADER_H
#define CPP_FUNCTIONS_HEADER_H

#include <TFile.h>
#include <TH1D.h>
#include <TH2F.h> // Correct type for histogram
#include <iostream>
#include <ROOT/RVec.hxx> // Include the RVec header
#include <string>        // Include string library

using namespace ROOT::VecOps; // Use the RVec namespace

namespace get_scale_factor {

class ScaleFactorHelper {
public:
    // Static method to get the instance of the class (Singleton pattern)
    static ScaleFactorHelper& getInstance(const std::string& year, bool is_data) {
        static ScaleFactorHelper instance(year, is_data);
        return instance;
    }

    std::pair<float, float> getScaleFactorAndError(
            float pt1, 
            float pt2, 
            float mass1, 
            float mass2, 
            float score1, 
            float score2,
            int ntop
        ) { // for method 3
        if (!histo) {
            std::cerr << "Histogram not initialized!" << std::endl;
            return {1.0, 0.0};
        }

        // Method 3
        int binX1 = histo->GetXaxis()->FindBin(pt1);
        int binX2 = histo->GetXaxis()->FindBin(pt2);

        int binY1 = histo->GetYaxis()->FindBin(mass1);
        int binY2 = histo->GetYaxis()->FindBin(mass2);

        float scale_factor_1 = histo->GetBinContent(binX1, binY1);
        float scale_factor_2 = histo->GetBinContent(binX2, binY2);

        float scale_error_1 = histo->GetBinError(binX1, binY1);
        float scale_error_2 = histo->GetBinError(binX2, binY2);

        if (ntop == 2) {
            // float combined_scale_factor = scale_factor_1 * scale_factor_2;
            // float combined_scale_error = combined_scale_factor * TMath::Sqrt(
            //     (scale_error_1 / scale_factor_1) * (scale_error_1 / scale_factor_1) +
            //     (scale_error_2 / scale_factor_2) * (scale_error_2 / scale_factor_2)
            // );
            // return {combined_scale_factor, combined_scale_error}; 

            float tot_fact_1 = scale_factor_1 / (1 - scale_factor_1);
            float tot_fact_2 = scale_factor_2 / (1 - scale_factor_2);

            float sigma_tot_fact_1 = scale_error_1 / std::pow(1 - scale_factor_1, 2);
            float sigma_tot_fact_2 = scale_error_2 / std::pow(1 - scale_factor_2, 2);

            if (score1 >= 0.5 && score2 < 0.5) { 
                return {tot_fact_2 / 2, sigma_tot_fact_2 * 2};
                // return {scale_factor_2, scale_error_2};
            } 
            else if (score1 < 0.5 && score2 >= 0.5) { 
                return {tot_fact_1 / 2, sigma_tot_fact_1 * 2};
                // return {scale_factor_1, scale_error_1};
            }
            // else {
            else if (score1 < 0.5 && score2 < 0.5) {
                float combined_scale_factor = tot_fact_1 * tot_fact_2;
                float combined_scale_error = std::sqrt(
                    std::pow(sigma_tot_fact_1 * tot_fact_2, 2) + std::pow(sigma_tot_fact_2 * tot_fact_1, 2)
                );
                return {combined_scale_factor, combined_scale_error};
            }
            // else{return {1.0, 0.0};}
        }
        else if (ntop == 1) {
            // return {scale_factor_1, scale_error_1};

            float tot_fact_1 = scale_factor_1 / (1 - scale_factor_1);
            float tot_fact_2 = scale_factor_2 / (1 - scale_factor_2);
            float combined_scale_factor = tot_fact_1 + tot_fact_2 + (tot_fact_1 * tot_fact_2);

            float sigma_F1 = scale_error_1 / std::pow(1 - scale_factor_1, 2);
            float sigma_F2 = scale_error_2 / std::pow(1 - scale_factor_2, 2);
        
            float combined_scale_error = std::sqrt(
                std::pow((1 - tot_fact_2) * sigma_F1, 2) +
                std::pow((1 - tot_fact_1) * sigma_F2, 2)
            );
            return {combined_scale_factor, combined_scale_error}; 
            // return {tot_fact_1, sigma_F1};
            // return {(tot_fact_1 * tot_fact_2) * 0.5, sigma_F1};
        }
        return {1.0, 0.0}; 
}

private:
    TFile* eff_file;
    TH2F* histo;

    // Private constructor to prevent instantiation
    ScaleFactorHelper(const std::string& year, bool is_data) {
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
        if (is_data) {
            histo = dynamic_cast<TH2F*>(eff_file->Get("efficiency_tagger_pTVSmass_data_1tag")); // for data
            if (histo) {
                std::cout << "Tagging efficiency histogram loaded: efficiency_tagger_pTVSmass_data_1tag" << std::endl;
            }
        } else {
            histo = dynamic_cast<TH2F*>(eff_file->Get("efficiency_tagger_pTVSmass_mc_1tag")); // for MC
            if (histo) {
                std::cout << "Tagging efficiency histogram loaded: efficiency_tagger_pTVSmass_mc_1tag" << std::endl;
            }
        }

        if (!histo) {
            std::cerr << "Histogram not found in file: " << file_path << std::endl;
            eff_file->Close();
            histo = nullptr;
            return;
        }
    }

    // Helper function to determine the correct file path based on the year
    std::string getFilePath(const std::string& year) {
        if (year == "2018") {
            // return "tagging_efficiency_for_bkg_estimation/2018/tagging_efficiency_from_on_Z_2J_1T_pTVSmass_merged_ee_mumu_poiss_unc_leading_subleading_merged.root";
            return "tagging_efficiency_for_bkg_estimation/2018/tagging_efficiency_from_on_Z_2J_1T_pTVSmass_merged_ee_mumu_poiss_unc_leading_subleading_merged_noHadrTop.root";
            // return "tagging_efficiency_for_bkg_estimation/2018/test_only_leading_ee.root";
            // return "tagging_efficiency_for_bkg_estimation/2018/test_leading_subleading_ee.root";
        } else if (year == "2017") {
            // return "tagging_efficiency_for_bkg_estimation/2017/tagging_efficiency_from_on_Z_2J_1T_pTVSmass_merged_ee_mumu_poiss_unc_leading_subleading_merged.root";
            return "tagging_efficiency_for_bkg_estimation/2017/tagging_efficiency_from_on_Z_2J_1T_pTVSmass_merged_ee_mumu_poiss_unc_leading_subleading_merged_noHadrTop.root";
        } else if (year == "2016preVFP") {
            return "tagging_efficiency_for_bkg_estimation/2016preVFP/tagging_efficiency_from_on_Z_2J_1T_pTVSmass_merged_ee_mumu_poiss_unc_leading_subleading_merged_noHadrTop.root";
        } else if (year == "2016") {
            return "tagging_efficiency_for_bkg_estimation/2016/tagging_efficiency_from_on_Z_2J_1T_pTVSmass_merged_ee_mumu_poiss_unc_leading_subleading_merged_noHadrTop.root";
        } else if (year == "2022") {
            return "tagging_efficiency_for_bkg_estimation/2022/tagging_efficiency_from_on_Z_2J_1T_pTVSmass_merged_ee_mumu_poiss_unc_leading_subleading_merged_noHadrTop.root";
        } else if (year == "2022EE") {
            return "tagging_efficiency_for_bkg_estimation/2022EE/tagging_efficiency_from_on_Z_2J_1T_pTVSmass_merged_ee_mumu_poiss_unc_leading_subleading_merged_noHadrTop.root";
        } else {
            std::cerr << "Year " << year << " is not recognized!" << std::endl;
            std::exit(EXIT_FAILURE);
        }
        // return "tagging_efficiency_for_bkg_estimation/all_years_Run2/tagging_efficiency_from_on_Z_2J_1T_pTVSmass_merged_ee_mumu_poiss_unc_leading_subleading_merged.root";
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
    const RVec<float>& object_mass,
    const RVec<float>& object_score, 
    const int& ntagged_object,
    const std::string& year, 
    bool is_data
    ) {
    if (object_pt.size() < 2) {
        return {1.0, 0.0};
    }

    ScaleFactorHelper& helper = ScaleFactorHelper::getInstance(year, is_data);
    return helper.getScaleFactorAndError(object_pt[0], object_pt[1], object_mass[0], object_mass[1], object_score[0], object_score[1], ntagged_object); 
}

} // namespace get_scale_factor

#endif // CPP_FUNCTIONS_HEADER_H
