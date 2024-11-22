#include <filesystem>
#include "tinyfold.h"

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
    std::unique_ptr<TinyFold<Weight_Q4>> tinyfold(load_tinyfold<Weight_Q4>(model_dir));

    // Make `output` directory if it doesn't exist
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
        std::string pdb = tinyfold->operator()(seq);
        std::ofstream out(out_filename);
        out << pdb;
        out.close();
        std::cout << "Saved to " << out_filename << std::endl;
    }
    return 0;
}
