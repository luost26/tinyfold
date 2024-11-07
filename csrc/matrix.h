#ifndef MATRIX_H
#define MATRIX_H

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

    ~matrix()
    {
        delete[] data;
    }

    T* operator()(int i, int j)
    {
        return &data[i * n_cols + j];
    }
};


template <typename T>
void matmul(matrix<T> &A, matrix<T> &B, matrix<T> &C)
{
    for (int i = 0; i < A.n_rows; i++)
    {
        for (int j = 0; j < B.n_cols; j++)
        {
            T sum = 0;
            for (int k = 0; k < A.n_cols; k++)
            {
                sum += A(i, k) * B(k, j);
            }
            *C(i, j) = sum;
        }
    }
}


template <typename T>
void matmul_add(matrix<T> &A, matrix<T> &B, matrix<T> &bias, matrix<T> &C)
{
    for (int i = 0; i < A.n_rows; i++)
    {
        for (int j = 0; j < B.n_cols; j++)
        {
            T sum = *bias(i, j);
            for (int k = 0; k < A.n_cols; k++)
            {
                sum += A(i, k) * B(k, j);
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


// Implement cout to print matrix
template <typename T>
std::ostream& operator<<(std::ostream &os, matrix<T> &A)
{
    for (int i = 0; i < A.n_rows; i++)
    {
        for (int j = 0; j < A.n_cols; j++)
        {
            os << *A(i, j) << " ";
        }
        os << "\n";
    }
    return os;
}

#endif
