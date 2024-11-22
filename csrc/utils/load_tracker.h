#ifndef UTILS_LOAD_TRACKER_H
#define UTILS_LOAD_TRACKER_H

#include <vector>
#include <string>
#include <omp.h>

static std::vector<std::string>* loaded_files = nullptr;

inline void init_load_tracker() {
    #pragma omp critical
    {
        if (loaded_files != nullptr) {
            delete loaded_files;
        }
        loaded_files = new std::vector<std::string>();
    }
}

inline void track_load(const std::string &path) {
    #pragma omp critical
    {
        if (loaded_files != nullptr) {
            loaded_files->push_back(path);
        }
    }
}

#endif // UTILS_LOAD_TRACKER_H