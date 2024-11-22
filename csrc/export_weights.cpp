#include <iostream>
#include <fstream>
#include <filesystem>
#include "tinyfold.h"
#include "utils/load_tracker.h"

#define COPYFILE(FROM,TO) { \
    std::ifstream src(FROM, std::ios::binary); \
    std::ofstream dst(TO, std::ios::binary); \
    dst << src.rdbuf(); \
}

int main(int argc, char **argv) {
    if (argc != 3) {
        std::cerr << "Usage: " << argv[0] << " <in_weights> <out_weights>" << std::endl;
        return 1;
    }
    init_load_tracker();
    std::string model_dir(argv[1]);
    std::unique_ptr<TinyFold<Weight_Q4>> tinyfold(load_tinyfold<Weight_Q4>(model_dir));

    std::string out_dir(argv[2]);
    for (auto &src_path : *loaded_files) {
        std::string src_dir = src_path.substr(0, src_path.find_last_of('/'));
        std::string dst_path = out_dir + "/" + src_path.substr(model_dir.size());
        std::string dst_dir = dst_path.substr(0, dst_path.find_last_of('/'));

        std::string src_cfg_path = src_dir + "/config.txt";
        std::string dst_cfg_path = dst_dir + "/config.txt";
        std::ifstream dst_cfg_file(dst_cfg_path);
        if (!dst_cfg_file.good()) {
            std::cout << src_cfg_path << " -> " << dst_cfg_path << std::endl;
            COPYFILE(src_cfg_path, dst_cfg_path);
        }

        std::filesystem::create_directories(dst_dir);
        std::ifstream dst_file(dst_path);
        if (!dst_file.good()) {
            std::cout << src_path << " -> " << dst_path << std::endl;
            COPYFILE(src_path, dst_path);
        }
    }
}