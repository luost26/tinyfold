#ifndef QMATRIX_H
#define QMATRIX_H

enum NumBits {
    Q4 = 4,
    Q8 = 8,
};

template <NumBits num_bits>
struct quantized_matrix {
    char *data;
    int group_size;
    float *scale;
    float *zero_point;
    int n_rows;
    int n_cols;
};

#endif // QMATRIX_H
