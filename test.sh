python ./scripts/generate_test_data/ipa.py

pushd ./build
cmake .. && make && ./main
popd
