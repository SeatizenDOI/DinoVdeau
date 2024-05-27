import argparse
import sys

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
    
def get_args():
    parser = argparse.ArgumentParser(description="DINOv2 Image Classification Training Script")
    parser.add_argument('--image_size', type=int, default=518, help='Image size for both dimensions')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training and evaluation')
    parser.add_argument('--num_train_epochs', type=int, default=150, help='Number of training epochs')
    parser.add_argument('--initial_learning_rate', type=float, default=0.001, help='Initial learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.0001, help='Weight decay for the optimizer')
    parser.add_argument('--early_stopping_patience', type=int, default=10, help='Early stopping patience')
    parser.add_argument('--patience_lr_scheduler', type=int, default=5, help='Patience for learning rate scheduler')
    parser.add_argument('--factor_lr_scheduler', type=float, default=0.1, help='Factor for learning rate scheduler')
    parser.add_argument('--model_name', type=str, default="facebook/dinov2-large", help="Model name or path")
    # freeze_flag to be True by default and explicitly settable to False
    parser.add_argument('--freeze_flag', type=str2bool, default=True, help='Flag to freeze model backbone, set to False to disable freezing')
    # data_aug_flag to be True by default and explicitly settable to False
    parser.add_argument('--data_aug_flag', type=str2bool, default=True, help='Flag to enable data augmentation, set to False to disable it')


    # Check if running in an interactive environment and adjust accordingly
    if any("jupyter" in arg for arg in sys.argv):
        args = parser.parse_args(args=[])
    else:
        args = parser.parse_args()
    
    return args
