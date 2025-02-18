# Multimodal Captions

## Running instructions

```bash
conda activate multimodal-captions
wandb login

# First, train the model
python -m src.train --wandb --epochs 10 --batch-size 64

# Then run inference with the trained model
python -m src.inference --checkpoint checkpoints/model_epoch_9.pt
```

Note: You need to train the model first to generate checkpoint files before running inference.

Optional: install nvtop to monitor system usage on linux gpu
```bash
add-apt-repository -y ppa:flexiondotorg/nvtop;apt install nvtop
```

## Environment setup

### Setup Instructions

1. [Install conda/miniconda](https://docs.anaconda.com/miniconda/install/)
2. Create the environment from `env/environment.yml`
```bash
conda env create -f env/environment.yml -y
conda activate multimodal-captions
# install pytorch via pytorch channel so that mps can be used on macos
conda install pytorch torchvision -c pytorch
conda deactivate # to exit
```

### Installing packages

```bash
# conda packages
conda install pytorch
# community packages
conda install -c conda-forge wandb
# Update environment files
./env/export_env.sh
```

### How I created it

#### Manual Approach

1. Create environment and export env info
```bash
conda create --name multimodal-captions python=3.10 -y
# install all the needed tools
conda env export > env/full_environment.yml
conda env export --from-history > env/installed_environment.yml
cp installed_environment.yml env/environment.yml
```

2. Then manually edit `environment.yml` by copying the channels from `full_environment.yml` into `installed_environment.yml`

#### Automated Approach

Run the `./env/export_env.sh` script after each install.
