# Olas Predict Benchmark

This repo is for testing the performance of Olas tools on historical prediction market data, before deploying them in real-time. Find out more about how to install and run the benchmark, and details on the dataset below. 

# Contents

- [🏗 Initial Setup](#-initial-setup)
- [🏛 Dataset](#-dataset)
- [🤖 Results](#-results)

## Initial Setup

```console
git clone https://github.com/valory-xyz/olas-predict-benchmark.git
cd olas-predict-benchmark

# create env and add openai api key
cp .env.sample .env

# download the benchmark data
mkdir benchmark/data
cd benchmark/data
git lfs install 
git clone https://huggingface.co/datasets/valory/autocast
cd ..

# clone the tools repo
git submodule add https://github.com/valory-xyz/mech.git

# set up env
poetry install
poetry shell

# run benchmark
poetry run benchmark/run_benchmark.py
```

## Dataset

We start with the Autocast [dataset](https://huggingface.co/datasets/valory/autocast) from the paper "[Forecasting Future World Events with Neural Networks](http://arxiv.org/abs/2206.15474)", and refine it further for the purpose of testing the performance of Olas mech prediction tools. The original and refined dataset is stored on [HuggingFace](https://huggingface.co/datasets/valory/autocast). 

The refined dataset files are:
- `autocast_questions_filtered.json` - a JSON subset of the initial autocast dataset.
- `autocast_questions_filtered.pkl` - a pickle file mapping URLs to the scraped documents of the filtered dataset.
- `retrieved_docs.pkl` - this contains all texts that were scraped.

The following notebooks are also available:
- `./benchmark/nbs/0. download_dataset` - download and explore the refined dataset from HuggingFace 
- `./benchmark/nbs/refined_autocast_dataset.ipynb` - details the refining procedure of the Autocast dataset for mech benchmarking

### Filtering Criteria

The need for refinement arises due to issues like dead URLs, paywalls, etc. The refinement process filters out and retains only "working" URLs:

- URLs returning non-200 HTTP status codes are filtered out.
- URLs from sites that are difficult to scrape like twitter, bloomberg.
- Links with less than 1000 words are removed.

Only samples with a minimum of 5 working URLs are retained. The maximum number of working source links is 20.

## Results

| Tool                            | Accuracy           | Correct | Total | Mean Tokens Used  | Mean Cost ($)   |
|---------------------------------|--------------------|---------|-------|-------------------|-----------------|
| claude-prediction-offline       | 0.75               | 219     | 292   | 779.2979452054794 | 0.00689010958904109   |
| claude-prediction-online        | 0.6991643454038997 | 251     | 359   | 1531.7214484679666| 0.013557348189415024  |
| prediction-online               | 0.6369426751592356 | 300     | 471   | 1244.4819532908705| 0.0013571125265392657 |
| prediction-offline              | 0.6365914786967418 | 254     | 399   | 580.390977443609  | 0.0006222556390977439 |
| prediction-online-summarized-info| 0.635             | 254     | 400   | 1008.8325         | 0.0011212074999999995 |
| prediction-offline-sme          | 0.6021276595744681 | 283     | 470   | 1189.7255319148935| 0.0013500234042553034 |
| prediction-online-sme           | 0.5583864118895966 | 263     | 471   | 1853.4670912951167| 0.002086902335456468  |
