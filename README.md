# Mamba Transformer
This repo contains a PyTorch implementation of the Mamba-Transformer model.

For more Deails see [my blog post](https://medium.com/gopenai/completing-the-circle-building-a-mambatransformer-from-scratch-f92a02381bc8) . 


## Train Usage

**NOTE** : Before running the script, make sure to move your `train` and `valid` data into the `data` folder.

To run the script with default values:

```bash
python train.py
```

### Customizing Training Parameters

You can override the default values by providing arguments:

```bash
python train.py --epochs 20 --learning_rate 0.001 --device cuda --clip_grad_norm 2.0 --lr_scheduler
```

### Breakdown of Arguments

- `--epochs 20`: Sets the number of training epochs to 20.
- `--learning_rate 0.001`: Sets the learning rate to 0.001.
- `--device cuda`: Uses the GPU for training.
- `--clip_grad_norm 2.0`: Clips gradients with a maximum norm of 2.0.
- `--lr_scheduler`: Enables the learning rate scheduler.

## Generate Text Usage

To run the script with default values:

```bash
python inference.py
```

### Customizing Generation Parameters

You can override the default values by providing arguments:

```bash
python inference.py --max_length 50 --num_return_sequences 5 --device cpu --prompt "نتا "
```

### Breakdown of Arguments

- `--max_length 50`: Sets the maximum length of the generated sequences to 50 tokens.
- `--num_return_sequences 5`: Generates 5 sequences.
- `--device cpu`: Uses the CPU for generation.
- `--prompt "نتا "`: Starts generation with the prompt `"نتا "`.
