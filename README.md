[![Python 3.7](https://img.shields.io/badge/python-3.7-blue.svg)](https://www.python.org/downloads/release/python-370/)


# Motion Forecasting for Autonomous Vehicles using the Argoverse Motion Forecasting Dataset
Official Argoverse Links:
1) [Argoverse-API](https://github.com/argoai/argoverse-api.git)
2) [Argoverse-Forecasting Baselines](https://github.com/jagjeet-singh/argoverse-forecasting)
3) [Datasets] (https://www.argoverse.org/data.html#download-link)

## Table of Contents

> If you have any questions, feel free to open a [GitHub issue](https://github.com/jagjeet-singh/argoverse-forecasting/issues) describing the problem.

- [Installation](#installation)
- [Usage](#usage)

---

## Installation

Requires Linux, git, and Python 3.6+

### 1) Setup Argoverse API
Download argoverse-api
```
git clone https://github.com/argoai/argoverse-api.git
pip install -e argoverse-api
```

For literals, install mypy 
```
pip install mypy
```

Download HD Maps 
```cd argoverse-api/
wget https://s3.amazonaws.com/argoai-argoverse/hd_maps.tar.gz
tar -xvf hd_maps.tar.gz
rm hd_maps.tar.gz
```
Your directory will look like this:
```
argoverse-api
    └── data_loading
    └── evaluation
    └── map_representation
    └── utils
    └── visualization
└── map_files
└── license
...
```

### 2) Download Deep-Prediction Repository
```
cd ~/
git clone https://github.com/sapan-ostic/deep_prediction.git
cd deep_prediction/
```

Install the packages mentioned in `requirements.txt`
```
pip install -r requirements.txt
```

### 2) Download Data
Argoverse provides both the full dataset, computed features and the sample version of the dataset. Head to [their website](https://www.argoverse.org/data.html#download-link) to see the download option.
Here, we use pre-trained features.
```
mkdir features
cd features/
```
To Download argoverse features run the following commands in the terminal:
```
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1tsMMMxwAQiixUYlc9cC4iaCCjUAdbzFT' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1tsMMMxwAQiixUYlc9cC4iaCCjUAdbzFT" -O forecasting_features_val.pkl && rm -rf /tmp/cookies.txt 

wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1kCUZHOvEN_HcN07QEeXRPh2tdCS7j0kc' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1kCUZHOvEN_HcN07QEeXRPh2tdCS7j0kc" -O forecasting_features_train.pkl && rm -rf /tmp/cookies.txt

wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1kBj2C08T1mEn1eT_q30Aoudjg9uTH89z' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1kBj2C08T1mEn1eT_q30Aoudjg9uTH89z" -O forecasting_features_test.pkl && rm -rf /tmp/cookies.txt
```
Download Forecasting Samples
```
cd ..
wget https://s3.amazonaws.com/argoai-argoverse/forecasting_sample_v1.1.tar.gz
tar -xvf forecasting_sample_v1.1.tar.gz
rm forecasting_sample_v1.1.tar.gz
```

## Usage

Running Motion Forecasting baselines has the below 3 components. The runtimes observed on a p2.8xlarge instance (8 NVIDIA K80 GPUs, 32 vCPUs and 488 GiB RAM) are also provided for each part:

### 1) Run forecasting baselines (`const_vel_train_test.py`, `nn_train_test.py`, `lstm_train_test.py`)

#### Constant Velocity:

```
$ python const_vel_train_test.py --test_features features/forecasting_features_test.pkl --obs_len 20 --pred_len 30 --traj_save_path saved_traj/const_vel.pkl
```

| Component | Mode | Runtime |
| --- | --- | --- |
| Constant Velocity (`const_vel_train_test.py`) | train+test | less than 1 min |


#### K-Nearest Neighbors:

Using Map prior:
```
$ python nn_train_test.py --train_features features/forecasting_features_train.pkl --val_features features/forecasting_features_val.pkl --test_features features/forecasting_features_test.pkl --use_map --use_delta --obs_len 20 --pred_len 30 --n_neigh 3 --model_path saved_models/nn_model_map_prior.pkl --traj_save_path saved_traj/nn_traj_map_prior.pkl
```

Neither map nor social:
```
$ python nn_train_test.py --train_features features/forecasting_features_train.pkl --val_features features/forecasting_features_val.pkl --test_features features/forecasting_features_test.pkl --normalize --use_delta --obs_len 20 --pred_len 30 --n_neigh 9 --model_path saved_models/nn_model_none.pkl --traj_save_path saved_traj/nn_traj_none.pkl
```

| Component | Mode | Baseline | Runtime |
| --- | --- | --- | --- |
| K-Nearest Neighbors (`nn_train_test.py`) | train+test | Map prior | 3.2 hrs |
| K-Nearest Neighbors (`nn_train_test.py`) | train+test | Niether map nor social | 0.5 hrs | 

#### LSTM:

Using Map prior:
```
$ python lstm_train_test.py --train_features features/forecasting_features_train.pkl --val_features features/forecasting_features_val.pkl --test_features features/forecasting_features_test.pkl --use_map --use_delta --obs_len 20 --pred_len 30 --model_path saved_models/lstm.pth.tar 
```

Using Social features:
```
$ python lstm_train_test.py --train_features features/forecasting_features_train.pkl --val_features features/forecasting_features_val.pkl --test_features features/forecasting_features_test.pkl --use_social --use_delta --normalize --obs_len 20 --pred_len 30 --model_path saved_models/lstm.pth.tar
```

Neither map nor social:
```
$ python lstm_train_test.py --train_features features/forecasting_features_train.pkl --val_features features/forecasting_features_val.pkl --test_features features/forecasting_features_test.pkl --use_delta --normalize --obs_len 20 --pred_len 30 --model_path saved_models/lstm.pth.tar
```

| Component | Mode | Baseline | Runtime |
| --- | --- | --- | --- |
| LSTM (`lstm_train_test.py`) | train | Map prior | 2 hrs |
| LSTM (`lstm_train_test.py`) | test | Map prior | 1.5 hrs |
| LSTM (`lstm_train_test.py`) | train | Social | 5.5 hrs |
| LSTM (`lstm_train_test.py`) | test | Social | 0.1 hr |
| LSTM (`lstm_train_test.py`) | train | Neither Social nor Map | 5.5 hrs |
| LSTM (`lstm_train_test.py`) | test | Neither Social nor Map | 0.1 hr |
---

Also tested on Google Cloud Platform:
1 NVIDIA K80 GPU,
4 vCPUs - 15GB RAM 

### 3) Metrics and visualization

#### Evaluation metrics

Here we compute the metric values for the given trajectories. Since ground truth trajectories for the test set have not been released, you can run the evaluation on the val set. If doing so, make sure you don't train any of the above baselines on val set.

Some examples:

Evaluating a baseline that didn't use map and allowing 6 guesses
```
python eval_forecasting_helper.py --metrics --gt <path/to/ground/truth/pkl/file> --forecast <path/to/forecasted/trajectories/pkl/file> --horizon 30 --obs_len 20 --miss_threshold 2 --features <path/to/test/features> --max_n_guesses 6
```

Evaluating a baseline that used map prior and allowing 1 guesses along each of the 9 centerlines
```
python eval_forecasting_helper.py --metrics --gt <path/to/ground/truth/pkl/file> --forecast <path/to/forecasted/trajectories/pkl/file> --horizon 30 --obs_len 20 --miss_threshold 2 --features <path/to/test/features> --n_guesses_cl 1 --n_cl 9 --max_neighbors_cl 3
```

Evaluating a K-NN baseline that can use map for pruning and allowing 6 guesses
```
python eval_forecasting_helper.py --metrics --gt <path/to/ground/truth/pkl/file> --forecast <path/to/forecasted/trajectories/pkl/file> --horizon 30 --obs_len 20 --miss_threshold 2 --features <path/to/test/features> --prune_n_guesses 6
```

It will print out something like
```
------------------------------------------------
Prediction Horizon : 30, Max #guesses (K): 1
------------------------------------------------
minADE: 3.533317191869932
minFDE: 7.887520305278937
DAC: 0.8857479236783845
Miss Rate: 0.9290787402582446
------------------------------------------------
```

#### Visualization

Here we visualize the forecasted trajectories

```
python eval_forecasting_helper.py --viz --gt <path/to/ground/truth/pkl/file> --forecast <path/to/forecasted/trajectories/pkl/file> --horizon 30 --obs_len 20 --features <path/to/test/features>
```
Some sample results are shown below

| | |
|:-------------------------:|:-------------------------:|
| ![](images/lane_change.png) | ![](images/map_for_reference_1.png) |
| ![](images/right_straight.png) | ![](images/map_for_reference_2.png) |
