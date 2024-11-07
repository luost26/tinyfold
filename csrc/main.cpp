#include <iostream>
#include "matrix.h"
#include "folding/ipa.h"

int main() {
    matrix<float> A(2, 3);
    std::cout << "A = \n" << A << std::endl;
    return 0;
}
