#ifndef MATRIX_H
#define MATRIX_H

#include <iostream>
#include <string>
#include <cstring>
#include <cmath>
#include "utils/load_tracker.h"

template <typename T>
struct matrix
{
    T *data;
    int n_rows;
    int n_cols;

    matrix(int n_rows, int n_cols) : n_rows(n_rows), n_cols(n_cols)
    {
        data = new T[n_rows * n_cols];
        std::memset(data, 0, n_rows * n_cols * sizeof(T));
    }

    matrix(T * data, int n_rows, int n_cols) : data(data), n_rows(n_rows), n_cols(n_cols) {}

    matrix(const matrix &A) : n_rows(A.n_rows), n_cols(A.n_cols)
    {
        // std::cerr << "Copy constructor: " << A.n_rows << " " << A.n_cols << std::endl;
        data = new T[n_rows * n_cols];
        for (int i = 0; i < n_rows * n_cols; i++)
        {
            data[i] = A.data[i];
        }
    }

    matrix(const std::string & path, int n_rows, int n_cols) {
        data = new T[n_rows * n_cols];
        this->n_rows = n_rows;
        this->n_cols = n_cols;
        load_(*this, path);
    }

    ~matrix()
    {
        delete[] data;
    }

    inline T* operator()(int i, int j)
    {
        return &data[i * n_cols + j];
    }

    inline const T* operator()(int i, int j) const
    {
        return &data[i * n_cols + j];
    }

    inline int numel() const {
        return n_rows * n_cols;
    }
};


template <bool transposed_B = false, bool residual = false, typename T>
void matmul(const matrix<T> &A, const matrix<T> &B, matrix<T> &C)
{
    if (&A == &C || &B == &C)
    {
        std::cerr << "Matrix multiplication cannot be done inplace" << std::endl;
        exit(1);
    }
    int I, J, K1, K2;
    if (transposed_B) {
        I = A.n_rows;
        J = B.n_rows;
        K1 = A.n_cols;
        K2 = B.n_cols;
    } else {
        I = A.n_rows;
        J = B.n_cols;
        K1 = A.n_cols;
        K2 = B.n_rows;
    }
    if (C.n_rows != I || C.n_cols != J || K1 != K2)
    {
        std::cerr << "Matrix dimensions do not match" << std::endl;
        exit(1);
    }

    #pragma omp parallel for
    for (int i = 0; i < C.n_rows; i++)
    {
        for (int j = 0; j < C.n_cols; j++)
        {
            T sum = 0;
            for (int k = 0; k < A.n_cols; k++)
            {
                if constexpr (transposed_B) {
                    sum += *A(i, k) * *B(j, k);
                } else {
                    sum += *A(i, k) * *B(k, j);
                }
            }
            if constexpr (residual) {
                *C(i, j) += sum;
            } else {
                *C(i, j) = sum;
            }
        }
    }
}


enum ActivationType {None, ReLU, GELU};

const float SQRT2 = sqrtf(2.0f);


template <bool transposed_B = false, bool is_bias_vector = false, bool residual = false, ActivationType act_type = None, typename T>
void matmul_add(const matrix<T> &A, const matrix<T> &B, const matrix<T> &bias, matrix<T> &C)
{
    if (&A == &C || &B == &C)
    {
        std::cerr << "Matrix multiplication cannot be done inplace" << std::endl;
        exit(1);
    }
    int I, J, K1, K2;
    if (transposed_B) {
        I = A.n_rows;
        J = B.n_rows;
        K1 = A.n_cols;
        K2 = B.n_cols;
    } else {
        I = A.n_rows;
        J = B.n_cols;
        K1 = A.n_cols;
        K2 = B.n_rows;
    }
    if (C.n_rows != I || C.n_cols != J || K1 != K2)
    {
        std::cerr << "Matrix dimensions do not match: ";
        std::cerr << "A(" << A.n_rows << ", " << A.n_cols << ") ";
        std::cerr << "B(" << B.n_rows << ", " << B.n_cols << ") ";
        std::cerr << "bias(" << bias.n_rows << ", " << bias.n_cols << ") ";
        std::cerr << "out(" << C.n_rows << ", " << C.n_cols << ") " << std::endl;
        exit(1);
    }
    if (is_bias_vector) {
        if (bias.n_cols != 1) {
            std::cerr << "Bias vector must have 1 column" << std::endl;
            exit(1);
        }
        if (bias.n_rows != C.n_cols) {
            std::cerr << "Bias vector must have the same number of rows as the output matrix's number of cols" << std::endl;
            exit(1);
        }
    }

    #pragma omp parallel for
    for (int i = 0; i < C.n_rows; i++)
    {
        for (int j = 0; j < C.n_cols; j++)
        {
            T sum = 0;
            if constexpr (is_bias_vector) {
                if constexpr (transposed_B) {
                    sum = *bias(j, 0);
                } else {
                    sum = *bias(i, 0);
                }
            } else {
                sum = *bias(i, j);
            }
            for (int k = 0; k < A.n_cols; k++)
            {
                if constexpr (transposed_B) {
                    sum += *A(i, k) * *B(j, k);
                } else {
                    sum += *A(i, k) * *B(k, j);
                }
            }
            if constexpr (act_type == ReLU) {
                sum = std::max((T)0, sum);
            } else if constexpr (act_type == GELU) {
                sum = sum * 0.5 * (1.0 + erf(sum / SQRT2));
            }

            if constexpr (residual) {
                *C(i, j) += sum;
            } else {
                *C(i, j) = sum;
            }
        }
    }
}

template <typename T>
void softmax_(matrix<T> &A) {
    for (int i = 0; i < A.n_rows; i++) {
        T max_val = *A(i, 0);
        for (int j = 0; j < A.n_cols; j++) {
            max_val = std::max(max_val, *A(i, j));
        }
        T sum = 0;
        for (int j = 0; j < A.n_cols; j++) {
            *A(i, j) = exp(*A(i, j) - max_val);
            sum += *A(i, j);
        }
        for (int j = 0; j < A.n_cols; j++) {
            *A(i, j) /= sum;
        }
    }
}


template <typename T>
inline void fill_(matrix<T> & A, T val) {
    for (int i = 0; i < A.n_rows; i++) {
        for (int j = 0; j < A.n_cols; j++) {
            *A(i, j) = val;
        }
    }
}


template <typename T>
inline void zero_(matrix<T> & A) {
    memset(A.data, 0, A.n_rows * A.n_cols * sizeof(T));
}


template <typename T>
inline void add_(matrix<T> &A, const matrix<T> &B) {
    for (int i = 0; i < A.n_rows; i++) {
        for (int j = 0; j < A.n_cols; j++) {
            *A(i, j) += *B(i, j);
        }
    }
}


template <typename T>
inline void sub_(matrix<T> &A, const matrix<T> &B) {
    for (int i = 0; i < A.n_rows; i++) {
        for (int j = 0; j < A.n_cols; j++) {
            *A(i, j) -= *B(i, j);
        }
    }
}


template <typename T>
inline void mul_(matrix<T> &A, T b) {
    for (int i = 0; i < A.n_rows; i++) {
        for (int j = 0; j < A.n_cols; j++) {
            *A(i, j) *= b;
        }
    }
}


template <typename T>
inline bool allclose(const matrix<T> &A, const matrix<T> &B, T atol = 1e-6, T * error_output = nullptr) {
    if (A.n_rows != B.n_rows || A.n_cols != B.n_cols) {
        return false;
    }
    T largest_error = 0;
    for (int i = 0; i < A.n_rows; i++) {
        for (int j = 0; j < A.n_cols; j++) {
            T error = std::abs(*A(i, j) - *B(i, j));
            if (error > largest_error || error != error) {
                largest_error = error;
            }
        }
    }
    if (error_output != nullptr) {
        *error_output = largest_error;
    }
    return largest_error <= atol;
}


template <typename T>
void load_(matrix<T> & A, const std::string & path) {
    FILE *f = fopen(path.c_str(), "rb");
    if (!f) {
        std::cerr << "Cannot open file " << path << std::endl;
        exit(1);
    }
    track_load(path);
    std::ignore = fread(A.data, sizeof(T), A.n_rows * A.n_cols, f);
    fclose(f);
}


template <typename T>
std::ostream& operator<<(std::ostream &os, const matrix<T> &A)
{

    #define PRINT_ROW_ELEMENTS(MAT,ROW,LOW,UP) { \
        for (int j = LOW; j < UP; j++) { \
            os << *MAT(ROW, j); \
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

#define TEST_ALLCLOSE(NAME,A,B,ATOL) { \
    float error; \
    if (!allclose(A, B, ATOL, &error)) { \
        std::cout << "Test " << NAME << ": FAIL, error = " << error << std::endl; \
        std::cout << "Expected:\n" << B << std::endl; \
        std::cout << "Got:\n" << A << std::endl; \
    } else { \
        std::cout << "Test " << NAME << ": PASS" << std::endl; \
    } \
}

#endif
