#### MIL test task

This repository contains a test task to MIL about autoencoders. There ara several types of architetures, which I implemented:
  * Base fully-connected autoencoder
  * Simple convolutional based autoencoder
  * Masked autoencoder (a.k.a. MAE)
  
This README clarifies the process of lauching my code and also contains some comments about implementation. All experiments may be found in [notebook](https://github.com/johanDDC/MIL_autoencoder/blob/master/notebooks/experiments.ipynb). Note, that notebook contains results of my experiment, and also may use libraries that which are not presented in requirements. This notebook is considered to be viewed, not launched.

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

  * `--type` --- The type of architecture you use;
  * `--download` --- whether you need to download dataset or not. Default is `False`;
  * `--nuw_workers` --- scecify the number of workers, that dataloader will use. Defaul value is `1`;
  * `--pretrained_path` --- path to file, where your pretrained model is locadet. It must be `.pth` file, which contains state dictionary of your autoencoder by field `model`. If argument is not specified, the encoder will be trained from the beggining;
  * `--mode` --- whether to `fine_tune` your model your during classifier training, or use `probing` whithour weights adjustment. This parameter also effector of classifier architecture: for `probing` it is just a single linear layer, wherere as for `fine_tune` it is two linear layers, with dropout and activation between them. Default value is `fine_tune`;
  * `--freeze` --- whether to freeze encoder parameters during classifier training. This parameter is usefull, if you want to train more complex classifier without changing weights. Default value is `True`;
  * `--num_epoches` --- number of epoches for classifier to train;
  
Training is spent on dataset CIFAR100.

### Results

For each architecture I trained several classifiers. Each calssifier has been trained for 40 epoches.
My experiments showed the following results:

|          | No pretrain | Smart classifier + fine tune | Only classifier | Linear + fine tune | Only linear |
|----------|-------------|------------------------------|-----------------|--------------------|-------------|
| FC+l1    | .186        | .200                         | .206            | **.245**           | .190        |
| FC+l2    | .187        | .193                         | .204            | **.235**           | .187        |
| Conv-192 | .169        | .168                         | .160            | **.198**           | .134        |
| Conv-256 | .191        | .191                         | .210            | **.214**           | .186        |
| MAE      | ---         | .151                         | .082            | **.219**           | .088        |
|          |             |                              |                 |                    |             |

### Discussion

Firstly I note, that I gained best classification results in case, where I add simple linear classifier over encoder embeddings. This is true for all architectures I tried.

Secondly, the best classification results showed pretrained fully-connected model with simple linear classifier. It might be kinda confusing, but I find the following explanation: I fixed process of classifier training, same simple setting for all architectures. If we take a look at plot of loss of CNN and MAE models, we may note, that loss continues to decrease. So, the classifier training process is very simple for convolution-based classifier, and absolutely not appropriate for MAE.

I also wanted to try MAE-like approach with convolutional model: mask the large ratio of image and train UNet-like architecture as an autoencoder. Something like this is described in [Pathak et al. 2016](https://arxiv.org/abs/1604.07379). Unfortunately, I didn't have enough time for that, because I wanted to train MAE, and it turned to be very slow process with my resources.
