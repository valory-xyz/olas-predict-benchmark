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

# add the tools repo
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

### Scraping Approach

We have also scraped the content of the filtered URLs using a variety of libraries:

- pypdf2 for those urls that are a pdf
- wikipediaapi for those that are wikipedia
- requests, readability-lxml and html2text for most others
- requests, beautifulsoup and html2text for bbc

## Results

| Tool                            | Accuracy           | Correct | Total | Mean Tokens Used  | Mean Cost ($)   |
|---------------------------------|--------------------|---------|-------|-------------------|-----------------|
| claude-prediction-offline       | 0.7201834862385321 | 157     | 218   | 779.4770642201835 | 0.006891669724770637  |
| claude-prediction-online        | 0.6600660066006601 | 200     | 303   | 1505.3135313531352| 0.013348171617161701  |
| prediction-online               | 0.676737160120846  | 224     | 331   | 1219.6918429003022| 0.001332990936555879  |
| prediction-offline              | 0.6599326599326599 | 196     | 297   | 579.6565656565657 | 0.000621023569023569  |
| prediction-online-summarized-info| 0.6209150326797386| 190     | 306   | 1008.4542483660131| 0.0011213790849673195 |
| prediction-offline-sme          | 0.599406528189911  | 202     | 337   | 1190.2017804154302| 0.0013518635014836643 |
| prediction-online-sme           | 0.5905044510385756 | 199     | 337   | 1834.919881305638 | 0.0020690207715133428 |

### Notes

Compared to previous version:
1. Accuracy for online tools improve by around 4 or 5%
2. For most tools, online is no longer worse than offline (a lot of people pointed out that this was weird before)
3. Claude is the exception. However, online seems to answer significantly more questions (maybe because it has more data): 303 in total vs 218.