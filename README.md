#### MIL test task

This repository contains a test task to MIL about autoencoders. There ara several types of architetures, which I implemented:
  * Base fully-connected autoencoder
  * Simple convolutional based autoencoder
  * Masked autoencoder (a.k.a. MAE)
  
This README clarifies the process of lauching my code and also contains some comments about implementation. All experiments may be found in [https://github.com/johanDDC/MIL_autoencoder/blob/master/notebooks/experiments.ipynb](notebook). Note, that notebook contains results of my experiment, and also may use libraries that which are not presented in requirements. This notebook is considered to be viewed, not launched.

### Instalation

All libraries, my implementation use are presented in `requirements.txt`. For more fast training you should also be abel to run your code on GPU. The following comand will install all the libraries you need:

`pip install -r requirements.txt`

### Pretrain

The following command will launch pretraining:

`python pretrain.py`

To pretrain your autoencoder you should previously choose architecture type. Comandline argument `--type` is responsible for that. You may use one of following flags: `fc`, `conv` and `mae. Also some extra commandline arguments are implemented:

  * `--download` --- whether you need to download dataset or not. Defaul value is `False`. In that case it is considered, that dataset is located by path `data/`;
  * `--nuw_workers` --- scecify the number of workers, that dataloader will use. Defaul value is `1`;
  
Pretraining is spent on dataset CIFAR100.

### Train

After pretraining stage you also may train classifier over encoder. For that purpose use the following command:

`python train.py`

There are also several commandline options:

  *`--type` --- The type of architecture you use;
  *`--download` --- whether you need to download dataset or not. Default is `False`;
  *`--nuw_workers` --- scecify the number of workers, that dataloader will use. Defaul value is `1`;
  *`--pretrained_path` --- path to file, where your pretrained model is locadet. It must be `.pth` file, which contains state dictionary of your autoencoder by field `model`. If argument is not specified, the encoder will be trained from the beggining;
  *`--mode` --- whether to `fine_tune` your model your during classifier training, or use `probing` whithour weights adjustment. This parameter also effector of classifier architecture: for `probing` it is just a single linear layer, wherere as for `fine_tune` it is two linear layers, with dropout and activation between them. Default value is `fine_tune`;
  *`--freeze` --- whether to freeze encoder parameters during classifier training;
