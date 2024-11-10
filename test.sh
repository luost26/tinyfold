python ./scripts/generate_test_data/ipa.py

pushd ./build
make && ./main
popd
