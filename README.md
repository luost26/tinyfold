# TinyFold

## Environment

```bash
conda env create --name tinyfold --file env.yml
conda activate tinyfold
pip install -e .
```

## Data

Download pre-trained model weights:
```bash
bash ./data/download_all.sh
```

## Build TinyFold

```bash
mkdir build
cd build
cmake ..
make
```

## Generate Unit Test Data

```bash
python ./scripts/generate_test_data.py <test-case-name>
```

## Apply AWQ (Optional)

### Generate Calibration Data

```bash
python ./scripts/create_testset.py
```

### Visualize Weights and Activations Before/After AWQ

```bash
python ./scripts/awq_visualization.py <--layer {visualized-layer} (optional)>
```

If `visualized-layer` is not provided, the qkv projection layers in the second Transformer layer `esm.layers.1` will be visualized in `data/output/awq`.

## Export ESMFold Parameters

```bash
python ./scripts/export_esmfold.py <--out {export-path} (optional)> <--awq (optional)>
```

If `export-path` is not provided, the weights will be exported to `data/esmfold_fp32`.\
Please first generate calibration data if you want to apply AWQ.

## Run TinyFold

```bash
cd build
./main ../data/esmfold_fp32
```
