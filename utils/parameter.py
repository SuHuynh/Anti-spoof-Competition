import argparse

def str2bool(v):
    return v.lower() in ('true')

def get_parameters():

    parser = argparse.ArgumentParser()

    # Training setting
    parser.add_argument('--batch_size', type=int, default=10)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--imsize', type=int, default=112)
    parser.add_argument('--w_loss', type=float, default=0.0)

    # using pretrained
    parser.add_argument('--pretrained_model', type=int, default=None)

    # Misc
    parser.add_argument('--train', type=str2bool, default=True)
    parser.add_argument('--parallel', type=str2bool, default=True)
    parser.add_argument('--total_epoch', type=int, default=20)

    # Path
    parser.add_argument('--image_path', type=str, default='./data')
    parser.add_argument('--log_path', type=str, default='./logs')
    parser.add_argument('--model_save_path', type=str, default='./saved_models')

    # Step size
    parser.add_argument('--log_step', type=int, default=100)
    parser.add_argument('--model_save_step', type=int, default=2000)
    parser.add_argument('--plot_loss_step', type=int, default=50)
    parser.add_argument('--alpha', type=float, default=0.95)

    return parser.parse_args()
