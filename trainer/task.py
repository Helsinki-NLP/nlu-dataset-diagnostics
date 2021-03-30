from argparse import ArgumentParser
from trainer import experiment


def get_args():
    """Define the task arguments with the default values.

    Returns:
        experiment parameters
    """
    parser = ArgumentParser(description='NLU Dataset Diagnostics')

    parser.add_argument('--batch_size',
                        type=int,
                        default=16)
    parser.add_argument('--epochs',
                        type=int,
                        default=2)
    parser.add_argument('--log_every',
                        type=int,
                        default=100)
    parser.add_argument('--learning_rate',
                        type=float,
                        default=0.00005)
    parser.add_argument('--fraction_of_train_data',
                        type=float,
                        default=1
                        )
    parser.add_argument('--seed',
                        type=int,
                        default=1234)
    parser.add_argument('--corrupt_train',
                        action='store_true',
                        dest='corrupt_train')
    parser.add_argument('--corrupt_test',
                        action='store_true',
                        dest='corrupt_test')
    parser.add_argument('--pos',
                        type=list,
                        default=['NOUN'])
    parser.add_argument('--weight-decay',
                        default=0,
                        type=float)
    parser.add_argument('--job-dir',
                        help='GCS location to export models')
    parser.add_argument('--model-name',
                        help='The name of your saved model',
                        default='model.pth')

    return parser.parse_args()


def main():
    """Setup / Start the experiment
    """
    args = get_args()
    experiment.run(args)


if __name__ == '__main__':
    main()
