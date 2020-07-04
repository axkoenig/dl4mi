from argparse import ArgumentParser


def parse_args():

    parser = ArgumentParser("Trains a classifier for COVID-19 detection")

    parser.add_argument("--data_root", type=str, default="./data", help="Data root directory")
    parser.add_argument("--dataset_dir", type=str, default="./dataset", help="Dataset root directory with txt files")
    parser.add_argument("--log_dir", type=str, default="./logs", help="Logging directory")
    parser.add_argument("--log_name", type=str, default="classifier", help="Name of logging session")
    parser.add_argument("--ae_pth", type=str, default="models/autoencoder_30_05_2020_16_09_52_bs16_ep40_tl0.0064.pth", help="Path of trained autoencoder")
    parser.add_argument("--num_workers", type=int, default=4, help="num_workers > 0 turns on multi-process data loading")
    parser.add_argument("--img_size", type=int, default=224, help="Spatial size of training images")
    parser.add_argument("--max_epochs", type=int, default=4, help="Number of maximum training epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size during training")
    parser.add_argument("--nc", type=int, default=3, help="Number of channels in the training images")
    parser.add_argument("--nfc", type=int, default=8, help="Number of feature maps in classifier")
    parser.add_argument("--lr", type=float, default=0.0002, help="Learning rate for optimizer")
    parser.add_argument("--beta1", type=float, default=0.9, help="Beta1 hyperparameter for Adam optimizer")
    parser.add_argument("--beta2", type=float, default=0.999, help="Beta2 hyperparameter for Adam optimizer")
    parser.add_argument("--gpus", type=int, default=0, help="Number of GPUs. Use 0 for CPU mode")
    parser.add_argument("--debug", type=bool, default=False, help="Debug mode")
    parser.add_argument("--nz", type=int, default=1024, help="Autoencoder param - Size of latent code")
    parser.add_argument("--nfe", type=int, default=32, help="Autoencoder param - Number of feature maps in encoder")
    parser.add_argument("--nfd", type=int, default=32, help="Autoencoder param - Number of feature maps in decoder")
    parser.add_argument("--aug_min_scale", type=float, default=0.75, help="Minimum scale arg for RandomResizedCrop")
    parser.add_argument("--aug_max_scale", type=float, default=1.0, help="Maximum scale arg for RandomResizedCrop")
    parser.add_argument("--aug_rot", type=float, default=5, help="Degrees arg for RandomRotation")
    parser.add_argument("--aug_bright", type=float, default=0.2, help="Brightness arg for ColorJitter")
    parser.add_argument("--aug_cont", type=float, default=0.1, help="Contrast arg for ColorJitter")
    parser.add_argument("--folds", type=int, default=10, help="How many folds to use for cross validation")
    parser.add_argument("--nb_sanity_val_steps", type=int, default=0, help="Number of sanity val steps")
    
    args = parser.parse_args()
    return args

# TODO CHECK IF WE ARE ACTUALLY PLOTTING THE CORRECT METRICS

# TODO run resnet on image directly 
# TODO run resnet on anomaly map
# TODO increase capability of autoencoder because full variability of healthy data needs to be captured! 
# TODO use Unet as autoencoder -> https://github.com/mateuszbuda/brain-segmentation-pytorch/blob/master/unet.py
# TODO why is my loss stagnating (run resnet classifier on image directly, should reduce loss)
# TODO maybe reduce learning rate
# TODO check if kfold can shuffle directly
# TODO fix training accuracy metrics 
# TODO think about loss function
# TODO maybe introduce thresholing to anomaly map 
# TODO why is the anomaly map not mostly black? visualise without normalization 
# TODO research if loss function weighting or rather dataset rebalancing (nll_loss = nn.CrossEntropyLoss(weight=torch.tensor([2., 2., 250.]).to('cuda')))
# TODO check if makes sense to use torch.abs(anomaly_map) https://github.com/chirag126/CoroNet/blob/728049e695c4efe0a11dc2a1282dc4f16af504f4/train_CIN.py#L91
# TODO check if images contain same info on all three channels or not
# TODO add test/accuracy --> tensorboard will format it nicely

# Differentiate from CoroNet
# - maybe use two autoencoders (normal, pneumonia) as feature extractors (use only encoder) and then classify directly & fine tune encoders jointly
# - check if SVM improves performance