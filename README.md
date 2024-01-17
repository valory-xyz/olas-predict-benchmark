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