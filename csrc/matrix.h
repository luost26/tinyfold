#ifndef MATRIX_H
#define MATRIX_H

#include <iostream>
#include <string>
#include <cstring>
#include <cmath>

template <typename T>
struct matrix
{
    T *data;
    int n_rows;
    int n_cols;

    matrix(int n_rows, int n_cols) : n_rows(n_rows), n_cols(n_cols)
    {
        data = new T[n_rows * n_cols];
    }

    matrix(T * data, int n_rows, int n_cols) : data(data), n_rows(n_rows), n_cols(n_cols) {}

    matrix(const matrix &A) : n_rows(A.n_rows), n_cols(A.n_cols)
    {
        std::cout << "Copy constructor: " << A.n_rows << " " << A.n_cols << std::endl;
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
};


template <bool transposed_B = false, bool residual = false, typename T>
void matmul(const matrix<T> &A, const matrix<T> &B, matrix<T> &C)
{
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

    for (int i = 0; i < C.n_rows; i++)
    {
        for (int j = 0; j < C.n_cols; j++)
        {
            T sum = 0;
            for (int k = 0; k < A.n_cols; k++)
            {
                if (transposed_B) {
                    sum += *A(i, k) * *B(j, k);
                } else {
                    sum += *A(i, k) * *B(k, j);
                }
            }
            if (residual) {
                *C(i, j) += sum;
            } else {
                *C(i, j) = sum;
            }
        }
    }
}


template <bool transposed_B = false, bool is_bias_vector = false, bool residual = false, typename T>
void matmul_add(const matrix<T> &A, const matrix<T> &B, const matrix<T> &bias, matrix<T> &C)
{
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

    for (int i = 0; i < C.n_rows; i++)
    {
        for (int j = 0; j < C.n_cols; j++)
        {
            T sum = 0;
            if (is_bias_vector) {
                if (transposed_B) {
                    sum = *bias(j, 0);
                } else {
                    sum = *bias(i, 0);
                }
            } else {
                sum = *bias(i, j);
            }
            for (int k = 0; k < A.n_cols; k++)
            {
                if (transposed_B) {
                    sum += *A(i, k) * *B(j, k);
                } else {
                    sum += *A(i, k) * *B(k, j);
                }
            }
            if (residual) {
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
            if (*A(i, j) > max_val) {
                max_val = *A(i, j);
            }
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
inline void sub_(matrix<T> &A, const matrix<T> &B) {
    for (int i = 0; i < A.n_rows; i++) {
        for (int j = 0; j < A.n_cols; j++) {
            *A(i, j) -= *B(i, j);
        }
    }
}


template <typename T>
inline bool allclose(const matrix<T> &A, const matrix<T> &B, T atol = 1e-6, T * error_output = nullptr) {
    if (A.n_rows != B.n_rows || A.n_cols != B.n_cols) {
        return false;
    }
    for (int i = 0; i < A.n_rows; i++) {
        for (int j = 0; j < A.n_cols; j++) {
            T error = std::abs(*A(i, j) - *B(i, j));
            if (error > atol) {
                if (error_output) {
                    *error_output = error;
                }
                return false;
            }
        }
    }
    return true;
}


template <typename T>
void load_(matrix<T> & A, const std::string & path) {
    FILE *f = fopen(path.c_str(), "rb");
    if (!f) {
        std::cerr << "Cannot open file " << path << std::endl;
        exit(1);
    }
    fread(A.data, sizeof(T), A.n_rows * A.n_cols, f);
}


template <typename T>
std::ostream& operator<<(std::ostream &os, const matrix<T> &A)
{
    os << "[";
    for (int i = 0; i < A.n_rows; i++)
    {
        os << "[";
        for (int j = 0; j < A.n_cols; j++)
        {
            os << *A(i, j);
            if (j < A.n_cols - 1) os << " ";
        }
        os << "]";
        if (i < A.n_rows - 1) os << "\n";
    }
    os << ", size=(" << A.n_rows << ", " << A.n_cols << ")]\n";
    return os;
}

#endif
