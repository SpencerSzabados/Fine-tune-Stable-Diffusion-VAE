
# Fine-tuning Stable Diffusions VAE
This repository contains the implementation details for fine-tuning [Stable Diffusion's](https://github.com/CompVis/stable-diffusion) Variational Auto-Encoder, as outlined in my accompanying project post [Web](https://spencerszabados.github.io/projects/2024-05-26-fine-tune-stable-diffusion-vae/).

## Summary
This repository contains scripts for fine-tuning stable diffusions variational auto-encoder on custom image datasets. All the defaults are configured for training at a resolution of 512x512x3 but can be easily modified to support custom resolutions.

## Usege
### Hardware requirements 
Training requires a CUDA capable device. Fine-tuning on 512x512x3 images with a batch size of 8 and microbatches of 4 requires 42GB of (combined) VRAM and takes ~20mins to run 1000 steps on one NVIDIA L40s. The VRAM usege can be approximately halved by removing the use of perceptual similarity loss [lpips](https://github.com/richzhang/PerceptualSimilarity) within 'fine-tune.py'.

### Environment setup 
All python package requirements are listed within the 'requirements.yaml' file. To build the conda environment, named 'sdvae', run the following command:

```
$ conda env create -f requirements.yml
```
Then activate the environment using 
```
$ conda activate sdvae
```

### General usage
All launch options can be configured within the 'args.sh' file. The provided 'fine-tune.sh' script accepts a variety of different launch options: 
```
  --workdir /YOUR_CHECKPOINT_PATH/
  --data_dir /DATASET_PATH/
  --image_size {int}
  --microbatch {int factor of --global_batch_size} #-1 disables this feature
  --global_batch_size {int}
  --lr {float}
  --ema_rate {sequence of floats}
  --weight_decay {float}
  --lr_anneal_steps {int}
  --log_interval {int}
  --save_interval {int}
  --total_training_steps {int}
  --augment {0, 1}
  --num_workers {num gpus}
  --fp16 {True, False}
  ...
```
These parameters should be changed within 'args.sh'. Some default working values are provided for the training parameters.

The model can then be launched using 'fine-tune.sh' as:
```
$ bash fine-tune.sh
```
which calls the main python script 'fine-tune.py'.


## Code attribution
The given implementation(s) is loosely based on the github repository of [cccntu](https://github.com/cccntu/fine-tune-models), with the dataloader based on that in [Consistency Models](https://github.com/openai/consistency_models).


## Citation
@misc{szabados:2024,
    author = {Szabados, Spencer},
    title = {Fine-tune Stable Diffusion VAE},
    year = {2024},
    howpublished = {\url{https://spencerszabados.github.io/projects/2024-05-26-fine-tune-stable-diffusion-vae/}},
    url = {https://github.com/SpencerSzabados/Fine-tune-Stable-Diffusion-VAE},
}

