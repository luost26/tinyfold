#include <iostream>
#include "matrix.h"
#include "folding/linear.h"
#include "folding/ipa.h"
#include "folding/ipa_test.h"

int main() {
    // matrix<float> A("../data/c_test/matrix.bin", 5, 5);
    // matrix<float> B(A);
    // matrix<float> bias(5, 5);
    // fill_(bias, 1.0f);
    // matrix<float> out(5, 5);
    // matmul_add(A, B, bias, out);
    // std::cout << "bias = \n" << bias << std::endl;
    
    // std::cout << "A = \n" << out << std::endl;

    test_ipa();

    return 0;
}
