You can train your own model by following these instructions:


## Train a model

Now you can train a new model by running the script:

```bash
python train.py
```

By default this will train a model on CMA best track dataset and ERA5 data, periodically saving checkpoint files to the folder named `model_save`. The training script has a number of command-line flags that you can use to configure the model architecture, hyperparameters, and input / output settings:

### Global Setting
- `--gpu_num`: The id of the GPU that you want to use. Default is '0'.
- `--dataset_name`: The id of the GPU that you want to use. Default is '1950_2019'.
- `--modal`: The data of Data_2d. Default is 'gph'.
- `--pi_pre_epoch`: The number of epoch that train the GC-Net in advance. Default is '5'.

### Optimization

- `--batch_size`: How many sequences to use in each minibatch during training. Default is 96.
- `--num_iterations`: Number of training iterations. Default is 5,000.
- `--num_epochs`: Number of training iterations. Default is 100+pi_pre_epoch.

### Model options
Our model consists of three components 1) Generator 2) Pooling Module 3) Discriminator. These flags control the architecture hyperparameters for both generator and discriminator.

- `--embedding_dim`: Integer giving the dimension for the embedding layer for input (x, y) coordinates. Default is 32.
- `--num_layers`: Number of layers in LSTM. We only support num_layers = 1.
- `--dropout`: Float value specifying the amount of dropout. Default is 0 (no dropout).
- `--batch_norm`: Boolean flag indicating if MLP has batch norm. Default is False.
- `--mlp_dim`: Integer giving the dimensions of the MLP. Default is 128.
We use the same mlp options across all three components of the model.

**Generator Options**: The generator takes as input all the trajectories for a given sequence and jointly predicts socially acceptable trajectories. These flags control architecture hyperparameters specific to the generator:
- `--encoder_h_dim_g`: Integer giving the dimensions of the hidden layer in the encoder. Default is 64.
- `--decoder_h_dim_g`: Integer giving the dimensions of the hidden layer in the decoder. Default is 64.
- `--noise_dim`: Integer tuple giving the dimensions of the noise added to the input of the decoder. Default is (16,).
- `--noise_type`: Type of noise to be added. We support two options "uniform" and "gaussian" noise. Default is "gaussian".
- `--noise_mix_type`: The added noise can either be the same across all pedestrians or we can have a different per person noise. We support two options "global" and "ped". Default value is "ped".
- `--clipping_threshold_g`: Float value indicating the threshold at which the gradients should be clipped. Default is 0.
- `--g_learning_rate`: Learning rate for the generator. Default is 1e-4.
- `--g_steps`: An iteration consists of g_steps forward backward pass on the generator. Default is 1.


**Discriminator Options**: These flags control architecture hyperparameters specific to the discriminator:
- `--d_type`: The discriminator can either treat each trajectory independently as described in the paper (option "local") or it can follow something very similar to the generator and pool the information across trajectories to determine if they are real/fake (option "global"). Default is "local".
- `--encoder_h_dim_d`:  Integer giving the dimensions of the hidden layer in the encoder. Default is 128.
- `--d_learning_rate`: Learning rate for the discriminator. Default is 1e-4.
- `--d_steps`: An iteration consists of d_steps forward backward pass on the generator. Default is 2.
- `--clipping_threshold_d`: Float value indicating the threshold at which the gradients should be clipped. Default is 2.0.

### Loss Options
- `--best_k`: The number of future potential TC tendencies that MGTCF predicts. Default is 6.

### Output Options
These flags control outputs from the training script:

- `--output_dir`: Directory to which checkpoints will be saved. Default is current directory.
- `--print_every`: Training losses are printed and recorded every `--print_every` iterations. Default is 5.
- `--timing`: If this flag is set to 1 then measure and print the time that each model component takes to execute.
- `--checkpoint_every`: Checkpoints are saved to disk every `--checkpoint_every` iterations. Default is 200. Each checkpoint contains a history of training losses, error metrics like ADE, FDE etc,  the current state of the generator, discriminators, and optimizers, as well as all other state information needed to resume training in case it is interrupted. We actually save two checkpoints: one with all information, and one without model parameters; the latter is much smaller, and is convenient for exploring the results of a large hyperparameter sweep without actually loading model parameters.
- `--checkpoint_name`: Base filename for saved checkpoints; default is 'checkpoint', so the filename for the checkpoint with model parameters will be 'checkpoint_with_model.pt' and the filename for the checkpoint without model parameters will be 'checkpoint_no_model.pt'.
- `--restore_from_checkpoint`: Default behavior is to start training from scratch, and overwrite the output checkpoint path if it already exists. If this flag is set to 1 then instead resume training from the output checkpoint file if it already exists. This is useful when running in an environment where jobs can be preempted.
- `--checkpoint_start_from`: Default behavior is to start training from scratch; if this flag is given then instead resume training from the specified checkpoint. This takes precedence over `--restore_from_checkpoint` if both are given.
- `--num_samples_check`: When calculating metrics on training dataset limit the number of samples you want to evaluate on to ensure checkpointing is fast for big datasets.
