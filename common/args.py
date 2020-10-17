from argparse import ArgumentParser


def parse_args():

    parser = ArgumentParser("Trains a classifier for COVID-19 detection")

    # logistics
    parser.add_argument("--data_root", type=str, default="../data", help="Data root directory")
    parser.add_argument("--dataset_dir", type=str, default="../dataset", help="Dataset root directory with txt files")
    parser.add_argument("--models_dir", type=str, default="../models", help="Name of models directory for saving")
    parser.add_argument("--log_dir", type=str, default="../logs", help="Logging directory")
    parser.add_argument("--log_name", type=str, default="test", help="Name of logging session")
    parser.add_argument("--pretrained_ae_pth", type=str, default="models/unet_bs16_ep50_05_07_2020_17_42_48.pth", help="Path of trained autoencoder (only used in 2_anomaly_detection/autoencoder.py)")
    parser.add_argument("--num_workers", type=int, default=4, help="num_workers > 0 turns on multi-process data loading")
    parser.add_argument("--debug", type=bool, default=False, help="Debug mode")
    parser.add_argument("--num_sanity_val_steps", type=int, default=0, help="Number of sanity val steps")
    parser.add_argument("--num_plots_per_epoch", type=int, default=5, help="Number of plots per training epoch")
    parser.add_argument("--gpus", type=int, default=0, help="Number of GPUs. Use 0 for CPU mode")

    # training parameters
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size during training")
    parser.add_argument("--folds", type=int, default=10, help="How many folds to use for cross validation")
    parser.add_argument("--epochs_per_fold", type=int, default=5, help="Number of training epochs in each fold")
    parser.add_argument("--max_epochs", type=int, default=10, help="Number of training epochs of healthy U-Net (only used in 2_anomaly_detection/autoencoder.py)")

    # architecture
    parser.add_argument("--img_size", type=int, default=224, help="Spatial size of training images")
    parser.add_argument("--nc", type=int, default=3, help="Number of channels in the training images")
    parser.add_argument("--nz", type=int, default=1024, help="Autoencoder param - Size of latent code")
    parser.add_argument("--nfe", type=int, default=32, help="Autoencoder param - Number of feature maps in encoder")
    parser.add_argument("--nfd", type=int, default=32, help="Autoencoder param - Number of feature maps in decoder")

    # optimization
    parser.add_argument("--lr", type=float, default=0.0002, help="Learning rate for optimizer")
    parser.add_argument("--beta1", type=float, default=0.9, help="Beta1 hyperparameter for Adam optimizer")
    parser.add_argument("--beta2", type=float, default=0.999, help="Beta2 hyperparameter for Adam optimizer")
    parser.add_argument("--alpha", type=float, default=0.5, help="Weight of reconstruction loss (only used in 3_multitask_learning/classifier.py)")

    # data augmentation
    parser.add_argument("--aug_min_scale", type=float, default=0.75, help="Minimum scale arg for RandomResizedCrop")
    parser.add_argument("--aug_max_scale", type=float, default=1.0, help="Maximum scale arg for RandomResizedCrop")
    parser.add_argument("--aug_rot", type=float, default=5, help="Degrees arg for RandomRotation")
    parser.add_argument("--aug_bright", type=float, default=0.2, help="Brightness arg for ColorJitter")
    parser.add_argument("--aug_cont", type=float, default=0.1, help="Contrast arg for ColorJitter")

    args = parser.parse_args()
    return args
