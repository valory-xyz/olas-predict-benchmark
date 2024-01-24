# OLAS Predict Benchmark

## Download the data 

The latest version of the [Autocast dataset can be downloaded here](https://people.eecs.berkeley.edu/~hendrycks/autocast.tar.gz)

## Install

```console
git clone https://github.com/valory-xyz/olas-predict-benchmark.git
cd olas-predict-benchmark

# create env and add openai api key
cp .env.sample .env

# link the data
mkdir data
cd data
ln -s ~/path/to/data/autocast .
cd ..

# set up env
cd benchmark
poetry install
poetry shell
```

## Run 

```console
poetry run benchmark/run_benchmark.py
```

## Notebook for Refining Autocast Dataset for Mech Benchmarking
Please see `./nbs.refined_autocast_dataset.ipynb` for details.

### Purpose
- This notebook is designed to refine the autocast dataset for mech benchmarking use cases.
- The need for refinement arises due to issues like dead URLs, paywalls, etc.
- The refinement process filters out and retains only "working" URLs.

### Filtering Criteria
- URLs returning non-200 HTTP status codes are filtered out.
- URLs containing certain keywords identified during manual checking are excluded.
- Links with less than 1000 words are removed.

### Dataset Specifications
- The final refined dataset will contain a minimum of 5 and a maximum of 20 source links.

### Storage
- The refined dataset is hosted on [HuggingFace](https://huggingface.co/datasets/valory/autocast/tree/main).
- There are two important files:
    - `olas_benchmark.json` - a JSON subset of the initial autocast dataset.
    - `olas_docs.pkl` - a pickle file mapping URLs to the retrieved documents.