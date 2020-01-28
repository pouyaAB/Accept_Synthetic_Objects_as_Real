# Accept Synthetic Objects as Real

## Dataset 
Please download the dataset from this [link](https://drive.google.com/file/d/1zo0DtuIjLWhFkpZdk9o-26tJ54iXg1pV/view?usp=sharing).

In the downloaded dataset, you can find two folders. Each contains demonstrations (images and commands) for one task. Each demonstration folder contains a demonstration performed by controlling the robot and another folder a human performed the same task by hand. Most of the demonstrations are recorded from 3 different angles using 3 cameras.

Each demonstration folder is named randomly and there exists a corresponding text file. The text file contains rows of commands sent to the robot by the demonstrator each row has its timestamp. The same timestamps are also used to name the corresponding images in the robot folder.

In the project, the class `DatasetController` is responsible for reading and managing the dataset. It can be used to return batches of images and their corresponding commands to be used by the architecture. After downloading the dataset you need to extract it somewhere and change the config (`local_config.py`) file accordingly.

```
- Dataset
    - 5001 (Picking up task)
        - dem1
            - human
                - camera-1
                - camera-2
                - camera-3
            - robot
                - camera-1
                - camera-2
                    - t1.jpg
                    - t2.jpg
                    ...
                    - tn.jpg
                - camera-3
        - dem2
        ...
        - demN

        dem1.txt
        dem2.txt
        ...
        demN.txt
        
    - 5002 (Pushing task)
        - dem1
        - dem2
        ...
        - demN
    
        dem1.txt
        dem2.txt
        ...
        demN.txt
    
```
# Prerequisites
- Numpy
- Chainer
- Pandas

# Config File
After Downloading the dataset and setting the dataset path in `local_config.py`. You can also change the other parameters in the config file as you like. Here is a brief description of the parameters.
```
# task to be used in the training process. These are the name of the folders for picking and pushing tasks.
config['tasks'] = ['5001', '5002']

# batch size to be used for the model
config['batch_size'] = 12

# sequence size to be used for the LSTMs
config['sequence_size'] = 4

# Number of columns in the commands txt file
config['csv_col_num'] = 10

# Number of channels in the image (RGB)
config['num_channels'] = 3

# Number of joint
config['output_size'] = 7

# hidden dimension of the LSTMs
config['hidden_dimension'] = 64

# Latent size of the VAE-GAN
config['latent_size'] = 64

# Number of Gaussians used in the Mixture Density Network
config['num_mixture'] = 32
```
# Code

## VAE-GAN
All the codes related to details of the Encoder, Generator, Discriminator can be found in the folder `autoencoders`. This folder contains some of the architectures we tried in our research. The file `tower.py` contains the final architecture for VAE-GAN components.

## Image Transformer
It is used by the `DatasetController` to apply a fixed homography on the images and transform them randomly for regularization purposes.

## GPU
In the file `gpu.py`, the user can specify the number of GPUs to be used and their id.

## Model Controller
The `Model Controller` is where the training happens. In this GitHub project, we have `model_e2e.py` and `model.py`. These two are mostly similar except the fact that `model_e2e` has the motor network as well and trains the whole network end-to-end.

## Main
The main file to run the whole training process is `nf_train_mean_cost.py`. This file will initialize a ModelController defined in another python file and call its train function.

## Latent from Encoder
This file `latent_from_model.py` can be used to feed the whole dataset to the encoder and save the latent variable to be later used to finetune the motor network.

## Train LSTM alone
The `train_lstm_alone.py` can be used to train only the motor network on the extracted latents. Note that you need to change the path to the latents inside the code.

## Testing on the robot
Testing on the robot can be done using the file `test_lstm_on_robot.py`. You need ROS installed and this [ros_teleoperate](https://github.com/pouyaAB/ros_teleoperate) package as well.

## Pretrained Models
You can download the pretrained models from [here](https://drive.google.com/file/d/1Wq-yNIqoTFaSJEZQkOVS6tlFRmD-jOzA/view?usp=sharing). Reconstruction of a set random images from the dataset can be viewed in the 'sample' folder.


# Citation
If this project helped you in your research, please consider citing us. Here is a BibTeX entry:

```
@article{abolghasemi2019accept,
  title={Accept Synthetic Objects as Real: End-to-End Training of Attentive Deep Visuomotor Policies for Manipulation in Clutter},
  author={Abolghasemi, Pooya and B{\"o}l{\"o}ni, Ladislau},
  journal={arXiv preprint arXiv:1909.11128},
  year={2019}
}
```

The paper can be found [here](https://arxiv.org/abs/1909.11128)



