#include <filesystem>
#include <cstdlib>
#include "tinyfold.h"


template <typename WeightType>
int main_loop(const std::string &model_dir, std::vector<std::string> &in_seqs) {
    std::unique_ptr<TinyFold<WeightType>> tinyfold(load_tinyfold<WeightType>(model_dir));
    std::filesystem::create_directory("output");

    std::string out_prefix = "";
    if (std::getenv("OUT_PREFIX") != nullptr) {
        out_prefix = std::string(std::getenv("OUT_PREFIX")) + "_";
    }
    for (int i = 0;;i ++) {
        std::string out_filename = "output/" + out_prefix + std::to_string(i).insert(0, 4 - std::to_string(i).length(), '0') + ".pdb";
        // If the output file already exists, skip
        if (std::filesystem::exists(out_filename)) {
            continue;
        }
        std::string seq;
        if (!in_seqs.empty()) {
            seq = in_seqs[0];
            in_seqs.erase(in_seqs.begin());
            std::cout << "SEQ: " << seq << std::endl;
        } else {
            std::cout << "SEQ: ";
            std::cin >> seq;
            if (seq.empty()) {
                std::cout << "Exiting..." << std::endl;
                break;
            }
        }
        std::string pdb = tinyfold->operator()(seq);
        if (pdb.empty()) {
            continue;
        }
        std::ofstream out(out_filename);
        out << pdb;
        out.close();
        std::cout << "Saved to " << out_filename << std::endl;
    }
    return 0;
}


/*
    Demo sequence:
        ASAWPEEKNYHQPAILNSSALRQIAEGTSISEMWQNDLQPLLIERYPGSPGSYAARQHIMQRIQRLQADWVLEIDTFLSQTPYGYRSFSNIISTLNPTAKRHLVLACHYDSKYFSHWNNRVFVGATDS
  */
int main(int argc, char **argv) {
    if (argc != 2 && argc != 3) {
        std::cerr << "Usage: " << argv[0] << " <model_dir> <optional:seq_path>" << std::endl;
        return 1;
    }
    std::string model_dir(argv[1]);
    std::vector<std::string> in_seqs;
    if (argc == 3) {
        std::ifstream in_seqs_file(argv[2]);
        if (!in_seqs_file.is_open()) {
            std::cerr << "Failed to open " << argv[2] << std::endl;
            return 1;
        }
        std::string seq;
        while (std::getline(in_seqs_file, seq)) {
            in_seqs.push_back(seq);
        }
    }

    if (std::getenv("FP32") != nullptr) {
        std::cout << "TinyFold: Using FP32 model" << std::endl;
        return main_loop<Weight_FP32>(model_dir, in_seqs);
    } else {
        std::cout << "TinyFold: Using Q4 (4-bit quantized) model" << std::endl;
        return main_loop<Weight_Q4>(model_dir, in_seqs);
    }
}
