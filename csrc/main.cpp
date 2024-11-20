#include <iostream>
#include "folding/ipa_test.h"
#include "folding/structure_module_test.h"
#include "folding/adaptor_test.h"
#include "folding/test.h"
#include "plm/transformer_test.h"
#include "plm/esm_test.h"

int main() {
    // test_ipa();
    // test_affine_quat_conversion();
    // test_structure_module();
    // test_adaptor();
    // test_folding();
    // test_transformer();
    test_esm_small();

    return 0;
}
