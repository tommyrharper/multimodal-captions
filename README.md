# Multimodal Captions

## Environment setup

### Setup Instructions

1. [Install conda/miniconda](https://docs.anaconda.com/miniconda/install/)
2. Create the environment from `environment.yml`
```bash
conda env create -f environment.yml
conda activate multimodal-captions
conda deactivate # to exit
```

### Installing packages

```bash
# conda packages
conda install pytorch
# community packages
conda install -c conda-forge wandb
# Update environment files
./export_env.sh
```

### How I created it

1. Create environment and export env info
```bash
conda create --name multimodal-captions python=3.10 -y
# install all the needed tools
conda env export > full_environment.yml
conda env export --from-history > installed_environment.yml
cp installed_environment.yml environment.yml
```

2. Then manually edit `environment.yml` by copying the channels from `full_environment.yml` into `installed_environment.yml`
