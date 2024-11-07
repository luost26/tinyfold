#include <iostream>
#include "matrix.h"
#include "folding/linear.h"
#include "folding/ipa.h"

int main() {
    // matrix<float> A("../data/c_test/matrix.bin", 5, 5);
    // matrix<float> B(A);
    // matrix<float> bias(5, 5);
    // fill_(bias, 1.0f);
    // matrix<float> out(5, 5);
    // matmul_add(A, B, bias, out);
    // std::cout << "bias = \n" << bias << std::endl;
    
    // std::cout << "A = \n" << out << std::endl;
    InvariantPointAttention ipa = *load_invariant_point_attention("../data/c_test/ipa");
    std::cout << ipa << std::endl;
    std::cout << ipa.linear_kv_points_weight << std::endl;
    return 0;
}
