# Olas Predict Benchmark

This repo is for testing the performance of Olas tools on historical prediction market data, before deploying them in real-time. Find out more about how to install and run the benchmark, and details on the dataset below. 

# Contents

- [🏗 Initial Setup](#-initial-setup)
- [🏛 Dataset](#-dataset)
- [🤖 Results](#-results)

## Initial Setup

```console
git clone --recurse-submodules https://github.com/valory-xyz/olas-predict-benchmark.git
cd olas-predict-benchmark

# create env and add openai api key
cp .env.sample .env

# download the benchmark data
mkdir benchmark/data
cd benchmark/data

# For linux users or Windows users using WSL you need to install first the library
sudo apt-get install git-lfs

git lfs install 
git clone https://huggingface.co/datasets/valory/autocast
cd ..

# set up env
poetry install
poetry shell

# run benchmark
poetry run benchmark/run_benchmark.py
```

## Troubleshooting
In case you face some error with the torch library of the type "No module named 'torch._C'" please uninstall the torch library and install it again.
You might face some errors of missing libraries on the virtual environment, please check the virtual environment is activated and install them.

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

You can check out the leaderboard on our [HuggingFace Space](https://huggingface.co/spaces/valory/olas-prediction-leaderboard). 

### Notes

Compared to previous version:
1. Accuracy for online tools improve by around 4 or 5%
2. For most tools, online is no longer worse than offline (a lot of people pointed out that this was weird before)
3. Claude is the exception. However, online seems to answer significantly more questions (maybe because it has more data): 303 in total vs 218.