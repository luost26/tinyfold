#include <iostream>
#include "folding/ipa_test.h"
#include "folding/structure_module_test.h"
#include "folding/adaptor_test.h"
#include "folding/test.h"

int main() {
    test_ipa();
    test_affine_quat_conversion();
    test_structure_module();
    test_adaptor();
    test_folding();

    return 0;
}
