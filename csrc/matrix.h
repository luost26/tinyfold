#ifndef MATRIX_H
#define MATRIX_H

#include <iostream>
#include <string>
#include <cstring>

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

    T* operator()(int i, int j)
    {
        return &data[i * n_cols + j];
    }

    const T* operator()(int i, int j) const
    {
        return &data[i * n_cols + j];
    }
};


template <bool transposed_B = false, typename T>
void matmul(const matrix<T> &A, const matrix<T> &B, matrix<T> &C)
{
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
            *C(i, j) = sum;
        }
    }
}


template <bool transposed_B = false, bool is_bias_vector = false, typename T>
void matmul_add(const matrix<T> &A, const matrix<T> &B, const matrix<T> &bias, matrix<T> &C)
{
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
            *C(i, j) = sum;
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
void fill_(matrix<T> & A, T val) {
    for (int i = 0; i < A.n_rows; i++) {
        for (int j = 0; j < A.n_cols; j++) {
            *A(i, j) = val;
        }
    }
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
