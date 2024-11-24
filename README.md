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

## Export ESMFold weight

```bash
python ./scripts/export_esmfold.py <optional:export-path>
```

If `export-path` is not provided, the weights will be exported to `data/esmfold_fp32`.

## Run TinyFold

```bash
cd build
./main ../data/esmfold_fp32
```
