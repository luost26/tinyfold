#include <iostream>
#include "matrix.h"
#include "folding/ipa.h"

int main() {
    matrix<float> A("../data/c_test/matrix.bin", 5, 5);
    std::cout << "A = \n" << A << std::endl;
    return 0;
}
