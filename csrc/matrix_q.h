#ifndef MATRIX_Q_H
#define MATRIX_Q_H

#include <iostream>
#include "matrix.h"
#include "utils/load_tracker.h"

enum NumBits {
    Q4 = 4,
    Q8 = 8,
};

template <NumBits num_bits, int block_size>
struct quantized_matrix {
    unsigned char *data;
    float *scales;
    int *zero_points;
    int n_rows;
    int n_cols;

    quantized_matrix(int n_rows, int n_cols):
        n_rows(n_rows), n_cols(n_cols)
    {
        int numel = n_rows * n_cols;
        if (n_cols % 2 != 0) {
            std::cerr << "Number of columns is not a multiple of 2" << std::endl;
            return;
        }
        if (numel % block_size != 0) {
            std::cerr << "Matrix size is not a multiple of block size" << std::endl;
            return;
        }
        if (block_size % 8 != 0) {
            std::cerr << "Block size is not a multiple of 8" << std::endl;
            return;
        }
        int num_blocks = numel / block_size;
        scales = new float[num_blocks];
        memset(scales, 0, num_blocks * sizeof(float));
        zero_points = new int[num_blocks];
        memset(zero_points, 0, num_blocks * sizeof(int));
        if constexpr (num_bits == Q4) {
            data = new unsigned char[numel / 2];
            memset(data, 0, numel / 2);
        } else if constexpr (num_bits == Q8) {
            data = new unsigned char[numel];
            memset(data, 0, numel);
        }
    }

    ~quantized_matrix() {
        delete[] data;
        delete[] scales;
        delete[] zero_points;
    }

    int numel() const {
        return n_rows * n_cols;
    }

    int data_size() const {
        if constexpr (num_bits == Q8) {
            return numel();
        } else if constexpr (num_bits == Q4) {
            return numel() / 2;
        }
    }

    int num_blocks() const {
        return (n_rows * n_cols) / block_size;
    }

    inline int group_index(int row, int col) const {
        return (row * n_cols + col) / block_size;
    }

    inline int elem_index(int row, int col) const {
        if constexpr (num_bits == Q8) {
            return (row * n_cols + col);
        } else if constexpr (num_bits == Q4) {
            return (row * n_cols + col) / 2;
        }
    }

    inline int quantized_value_at(int row, int col) const {
        if constexpr (num_bits == Q8) {
            return data[row * n_cols + col];
        } else if constexpr (num_bits == Q4) {
            int elem_idx = row * n_cols + col;
            return (elem_idx % 2 == 0) ? (data[elem_idx / 2] >> 4) : (data[elem_idx / 2] & 0x0F);
        }
    }

    inline float dequantize(int row, int col) const {
        if constexpr (num_bits == Q8) {
            int w_q = data[row * n_cols + col];
            int grp_idx = group_index(row, col);
            return scales[grp_idx] * (w_q - zero_points[grp_idx]);
        } else if constexpr (num_bits == Q4) {
            int elem_idx = row * n_cols + col;
            int w_q = (elem_idx % 2 == 0) ? (data[elem_idx / 2] >> 4) : (data[elem_idx / 2] & 0x0F);
            int grp_idx = group_index(row, col);
            return scales[grp_idx] * (w_q - zero_points[grp_idx]);
        }
    }

    inline void dequantize(matrix<float> &out) const {
        if (out.n_rows != n_rows || out.n_cols != n_cols) {
            std::cerr << "Output matrix size mismatch" << std::endl;
            return;
        }
        for (int i = 0; i < n_rows; i ++) {
            for (int j = 0; j < n_cols; j ++) {
                *out(i, j) = dequantize(i, j);
            }
        }
    }

    inline void dequantize_q4(int row, int col, float &v1, float &v2) const {
        if constexpr (num_bits != Q4) {
            std::cerr << "dequantize_q4 is only available for Q4 quantized matrices" << std::endl;
            return;
        }
        int elem_idx = row * n_cols + col;
        int grp_idx = group_index(row, col);
        int w_q = data[elem_idx / 2];
        float scale = scales[grp_idx];
        int zero_point = zero_points[grp_idx];
        v1 = scale * ((w_q >> 4) - zero_point);
        v2 = scale * ((w_q & 0x0F) - zero_point);
    }
};

template <NumBits num_bits, int block_size>
quantized_matrix<num_bits, block_size>* quantize(const matrix<float> &A, quantized_matrix<num_bits, block_size> *out = nullptr) {
    const int numel = A.numel();
    if (numel % block_size != 0) {
        std::cerr << "Matrix size is not a multiple of block size" << std::endl;
        return nullptr;
    }
    const int num_blocks = A.numel() / block_size;
    
    constexpr int max_int = (1 << (int)num_bits) - 1;
    quantized_matrix<num_bits, block_size> *q_matrix = out;

    if (q_matrix == nullptr) {
        q_matrix = new quantized_matrix<num_bits, block_size>(A.n_rows, A.n_cols);
    } else {
        if (q_matrix->n_rows != A.n_rows || q_matrix->n_cols != A.n_cols) {
            std::cerr << "Output matrix size mismatch" << std::endl;
            exit(1);
        }
    }

    #pragma omp parallel for
    for (int block_idx = 0; block_idx < num_blocks; block_idx ++) {
        float min_val = std::numeric_limits<float>::max();
        float max_val = std::numeric_limits<float>::min();
        for (int i = 0; i < block_size; i ++) {
            float val = A.data[block_idx * block_size + i];
            min_val = std::min(min_val, val);
            max_val = std::max(max_val, val);
        }
        float scale = std::max(max_val - min_val, 1e-5f) / max_int;
        int zero_point = std::min(std::max(-(int)std::round(min_val / scale), 0), max_int);

        q_matrix->scales[block_idx] = scale;
        q_matrix->zero_points[block_idx] = zero_point;

        if constexpr (num_bits == Q8) {
            for (int i = 0; i < block_size; i ++) {
                float w = A.data[block_idx * block_size + i];
                int w_q = (int)std::round(w / scale) + zero_point;
                w_q = std::min(std::max(w_q, 0), max_int);
                q_matrix->data[block_idx * block_size + i] = w_q;
            }
        } else if constexpr (num_bits == Q4) {
            for (int i = 0; i < block_size; i += 2) {
                float w1 = A.data[block_idx * block_size + i];
                float w2 = A.data[block_idx * block_size + i + 1];
                int w1_q = (int)std::round(w1 / scale) + zero_point;
                w1_q = std::min(std::max(w1_q, 0), max_int);
                int w2_q = (int)std::round(w2 / scale) + zero_point;
                w2_q = std::min(std::max(w2_q, 0), max_int);
                q_matrix->data[(block_idx * block_size + i) / 2] = (w1_q << 4) | w2_q;
            }
        }
    }
    return q_matrix;
}


const std::string QMATRIX_META_SUFFIX = ".qmeta";

template <NumBits num_bits, int block_size>
void load_(quantized_matrix<num_bits, block_size> & A, const std::string & metadata_path) {
    const std::string path_trunk = metadata_path.substr(0, metadata_path.size() - 6);  // Suffix: .qmeta
    const std::string data_path = path_trunk + ".data";
    const std::string scales_path = path_trunk + ".scales";
    const std::string zero_points_path = path_trunk + ".zero_points";

    int m_num_bits, m_block_size, m_n_rows, m_n_cols;
    std::ifstream meta_file(metadata_path);
    meta_file >> m_num_bits >> m_block_size >> m_n_rows >> m_n_cols;
    if (m_num_bits != num_bits || m_block_size != block_size || m_n_rows != A.n_rows || m_n_cols != A.n_cols) {
        std::cerr << "Metadata mismatch" << std::endl;
        exit(1);
    }

    std::ifstream data_file(data_path, std::ios::binary);
    data_file.read((char*)A.data, A.data_size());

    std::ifstream scales_file(scales_path, std::ios::binary);
    scales_file.read((char*)A.scales, A.num_blocks() * sizeof(float));

    std::ifstream zero_points_file(zero_points_path, std::ios::binary);
    zero_points_file.read((char*)A.zero_points, A.num_blocks() * sizeof(int));

    track_load(metadata_path);
    track_load(data_path);
    track_load(scales_path);
    track_load(zero_points_path);
}

template <NumBits num_bits, int block_size>
void save_(const quantized_matrix<num_bits, block_size> & A, const std::string & metadata_path) {
    const std::string path_trunk = metadata_path.substr(0, metadata_path.size() - 6);  // Suffix: .qmeta
    const std::string data_path = path_trunk + ".data";
    const std::string scales_path = path_trunk + ".scales";
    const std::string zero_points_path = path_trunk + ".zero_points";

    std::ofstream meta_file(metadata_path);
    meta_file << num_bits << " " << block_size << " " << A.n_rows << " " << A.n_cols << std::endl;

    std::ofstream data_file(data_path, std::ios::binary);
    data_file.write((char*)A.data, A.data_size());

    std::ofstream scales_file(scales_path, std::ios::binary);
    scales_file.write((char*)A.scales, A.num_blocks() * sizeof(float));

    std::ofstream zero_points_file(zero_points_path, std::ios::binary);
    zero_points_file.write((char*)A.zero_points, A.num_blocks() * sizeof(int));
}

template <NumBits num_bits, int block_size>
inline void mul_(quantized_matrix<num_bits, block_size> &A, float b) {
    int num_blocks = A.num_blocks();
    for (int i = 0; i < num_blocks; i++) {
        A.scales[i] *= b;
    }
}

template <NumBits num_bits, int block_size>
std::ostream& operator<<(std::ostream &os, const quantized_matrix<num_bits, block_size> &A)
{

    #define PRINT_ROW_ELEMENTS(MAT,ROW,LOW,UP) { \
        for (int j = LOW; j < UP; j++) { \
            os << MAT.dequantize(ROW, j) << "(" << MAT.quantized_value_at(ROW, j) << ")"; \
            if (j < UP - 1) os << " "; \
        } \
    }

    #define PRINT_ROW(MAT,ROW,NUM_COLS) { \
        os << "["; \
        if (NUM_COLS <= 10) { \
            PRINT_ROW_ELEMENTS(MAT,ROW,0,NUM_COLS); \
        } else { \
            PRINT_ROW_ELEMENTS(MAT,ROW,0,5); \
            os << "... "; \
            PRINT_ROW_ELEMENTS(MAT,ROW,NUM_COLS-5,NUM_COLS); \
        } \
        os << "]\n"; \
    }

    os << "[";
    if (A.n_rows <= 10) {
        for (int i = 0; i < A.n_rows; i++) {
            PRINT_ROW(A,i,A.n_cols);
        }
    } else {
        for (int i = 0; i < 5; i++) {
            PRINT_ROW(A,i,A.n_cols);
        }
        os << "...\n";
        for (int i = A.n_rows-5; i < A.n_rows; i++) {
            PRINT_ROW(A,i,A.n_cols);
        }
    }
    #undef PRINT_ROW_ELEMENTS
    #undef PRINT_ROW
    os << ", size=(" << A.n_rows << ", " << A.n_cols << ")]\n";
    return os;
}

#endif // MATRIX_Q_H
