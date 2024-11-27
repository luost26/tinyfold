#include <filesystem>
#include <cstdlib>
#include "tinyfold.h"


template <typename WeightType>
int main_loop(const std::string &model_dir) {
    std::unique_ptr<TinyFold<WeightType>> tinyfold(load_tinyfold<WeightType>(model_dir));
    std::filesystem::create_directory("output");

    for (int i = 0;;i ++) {
        std::string out_filename = "output/" + std::to_string(i).insert(0, 4 - std::to_string(i).length(), '0') + ".pdb";
        // If the output file already exists, skip
        if (std::filesystem::exists(out_filename)) {
            continue;
        }
        std::string seq;
        std::cout << "SEQ: ";
        std::cin >> seq;
        if (seq.empty()) {
            std::cout << "Exiting..." << std::endl;
            break;
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
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <model_dir>" << std::endl;
        return 1;
    }
    std::string model_dir(argv[1]);

    if (std::getenv("FP32") != nullptr) {
        std::cout << "TinyFold: Using FP32 model" << std::endl;
        return main_loop<Weight_FP32>(model_dir);
    } else {
        std::cout << "TinyFold: Using Q4 (4-bit quantized) model" << std::endl;
        return main_loop<Weight_Q4>(model_dir);
    }
}
