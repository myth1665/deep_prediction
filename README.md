# Social GAN for Autonomous Vehicle Motion Forecasting

The origin code for Social GAN provided by Agrim Gupta et.al. has been modified with data preprocessing and integration of argoverse-api for the argoverse dataset.

**<a href="https://arxiv.org/abs/1803.10892">Social GAN: Socially Acceptable Trajectories with Generative Adversarial Networks</a>**

A better understanding of agents' behaviour in a dynamic traffic environment is required for an efficient modelling and navigation of autonomous vehicles. In this project we plan to address the problem of motion forecasting of traffic actors through experimentation on the Argoverse Motion Forecasting dataset. We attempt to tackle this challenge using Generative Adversarial Networks (GANs) and compare out results with baseline methods of seq-to-seq prediction and social LSTM provided by the Argoverse Challenge.

Below we show an examples of predictions made by our model in complex scenarios. Each traffic actor category is denoted by a different color. 

<div align='center'>
<img src="images/2.gif"></img>
<img src="images/3.gif"></img>
</div>

## Model
Our model consists of three key components: Generator (G), Pooling Module (PM) and Discriminator (D). G is based on encoder-decoder framework where we link the hidden states of encoder and decoder via PM. G takes as input trajectories of all people involved in a scene and outputs corresponding predicted trajectories. D inputs the entire sequence comprising both input trajectory and future prediction and classifies them as “real/fake”.

<div align='center'>
  <img src='images/model.png' width='1000px'>
</div>

## Setup
All code was developed and tested on Ubuntu 16.04 with Python 3.5 and PyTorch 0.4.

You can setup a virtual environment to run the code like this:

```bash
python3 -m venv env               # Create a virtual environment
source env/bin/activate           # Activate virtual environment
pip install -r requirements.txt   # Install dependencies
echo $PWD > env/lib/python3.5/site-packages/sgan.pth  # Add current directory to python path
# Work for a while ...
deactivate  # Exit virtual environment
```

## Pretrained Models
You can download pretrained models by running the script `bash scripts/download_models.sh`. This will download the following models:

- `sgan-models/<dataset_name>_<pred_len>.pt`: Contains 10 pretrained models for all five datasets. These models correspond to SGAN-20V-20 in Table 1.
- `sgan-p-models/<dataset_name>_<pred_len>.pt`: Contains 10 pretrained models for all five datasets. These models correspond to SGAN-20VP-20 in Table 1.

Please refer to [Model Zoo](MODEL_ZOO.md) for results.

## Running Models
You can use the script `scripts/evaluate_model.py` to easily run any of the pretrained models on any of the datsets. For example you can replicate the Table 1 results for all datasets for SGAN-20V-20 like this:

```bash
python scripts/evaluate_model.py \
  --model_path models/sgan-models
```

## Training new models
Instructions for training new models can be [found here](TRAINING.md).
