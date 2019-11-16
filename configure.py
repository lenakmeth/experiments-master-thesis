import argparse
import sys
import torch


def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser()

    # Find if CUDA is used
    parser.add_argument("--USE_CUDA", default=False, type=bool, help="Use CUDA?")
    # Train
    parser.add_argument("--training", default=True, type=bool, help="Train or not?")

    # Data parameters
    parser.add_argument("--PAD_TOKEN", default=0, type=int, help="PAD token")
    parser.add_argument(
        "--SOS_TOKEN", default=1, type=int, help="Start of sequence token"
    )
    parser.add_argument(
        "--EOS_TOKEN", default=2, type=int, help="End of sequence token"
    )
    parser.add_argument("--UNK_TOKEN", default=3, type=int, help="Unknown token")

    # Model Hyper-parameters
    # Configure models
    parser.add_argument(
        "--attn_model", default="dot", type=str, help="Options: dot/general/concat"
    )
    parser.add_argument(
        "--hidden_size",
        default=100,
        type=int,
        help="Dimensionality of RNN hidden (default: 100)",
    )
    parser.add_argument(
        "--embed_size",
        default=300,
        type=int,
        help="Dimensionality of char embedding (default: 300)",
    )
    parser.add_argument("--n_layers", default=1, type=int, help="Number of layers")
    parser.add_argument("--dropout", default=0.1, type=int, help="Dropout probability")
    parser.add_argument(
        "--batch_size", default=20, type=int, help="Batch Size (default: 20)"
    )
    parser.add_argument(
        "--checkpoint_dir",
        default="checkpoints",
        type=str,
        help="Checkpoint directory from training run",
    )

    # Configure training/optimization

    parser.add_argument("--lang", default="", type=str, help="Language.")
    parser.add_argument(
        "--n_epochs",
        default=20,
        type=int,
        help="Number of training epochs (Default: 20)",
    )
    parser.add_argument("--clip", default=50.0, type=float, help="Grad clip.")
    parser.add_argument("--teacher_forcing_ratio", default=0.5, type=float, help=" ")
    parser.add_argument("--decoder_learning_ratio", default=5.0, type=float, help=" ")

    #    # Misc
    #    parser.add_argument("--desc", default = "",
    #                        type = str, help = "Description for model")
    #    parser.add_argument("--dropout_keep_prob", default = 0.5,
    #                        type = float, help = "Dropout keep probability of output layer (default: 0.5)")
    #    parser.add_argument("--l2_reg_lambda", default = 1e-5,
    #                        type = float, help = "L2 regularization lambda (default: 1e-5)")
    #
    #    # Training parameters
    #    parser.add_argument("--display_every", default = 10,
    #                        type = int, help = "Number of iterations to display training information")
    #    parser.add_argument("--evaluate_every", default = 100,
    #                        type = int, help = "Evaluate model on dev set after this many steps (default: 100)")
    #    parser.add_argument("--num_checkpoints", default = 5,
    #                        type = int, help = "Number of checkpoints to store (default: 5)")
    #
    #    parser.add_argument("--decay_rate", default = 0.9,
    #                        type = float, help = "Decay rate for learning rate (Default: 0.9)")
    #
    #    # Testing parameters
    #    # Misc Parameters
    #    parser.add_argument("--allow_soft_placement", default = True,
    #                        type = bool, help = "Allow device soft device placement")
    #    parser.add_argument("--log_device_placement", default = False,
    #                        type = bool, help = "Log placement of ops on devices")
    #    parser.add_argument("--gpu_allow_growth", default = True,
    #                        type = bool, help = "Allow gpu memory growth")

    if len(sys.argv) == 0:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()

    return args
