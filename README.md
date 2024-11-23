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

## Generate Unit Test Data

```bash
python ./scripts/generate_test_data.py <test-case-name>
```

## Export ESMFold weight

```bash
python ./scripts/generate_test_data.py esmfold
```

Afterwards, the exported weights can be found in `./data/c_test/esmfold`
