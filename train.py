import argparse


if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument("-t", "--type", required=True, type=str, help="autoencoder architecture type")
    args.add_argument("-d", "--download", default=False, type=bool, help="download dataset")
    args.add_argument("-nw", "--num_workers", default=1, type=int, help="num workers for dataloader")
    args.add_argument("-pp", "--pretrained_path", default=None, type=str, help="Path to pretrained model")
    args.add_argument("-m", "--mode", default="probing", type=str, help="Mode: fine tuning or probing")
    args.add_argument("-nopt", "--no_pretrained", default=False, type=bool, help="Train encoder from the start")
    args = args.parse_args()

    path = args.pretrained_path
