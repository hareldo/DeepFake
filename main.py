"""Main training script."""
import argparse

from torch import nn
from torch import optim

from utils import load_dataset, load_model, choose_criterion
from trainer import LoggingParameters, Trainer


# Arguments
def parse_args():
    """Parse script arguments.

    Get training hyper-parameters such as: learning rate, momentum,
    batch size, number of training epochs and optimizer.
    Get training dataset and the model name.
    """
    parser = argparse.ArgumentParser(description='Training models with Pytorch')
    parser.add_argument('--lr', default=0.001, type=float,
                        help='learning rate')
    parser.add_argument('--momentum', default=0.9, type=float,
                        help='SGD momentum')
    parser.add_argument('--batch_size', '-b', default=16, type=int,
                        help='Training batch size')
    parser.add_argument('--epochs', '-e', default=15, type=int,
                        help='Number of epochs to run')
    parser.add_argument('--model', '-m', default='Origin', type=str,
                        help='Model name: Origin (Efficientnet) or '
                             'Triplet (Efficientnet with embedding and tripplet loss)')
    parser.add_argument('--embedding_dimension', '-ed', default=512, type=int,
                        help="Dimension of the embedding vector (default: 512)")
    parser.add_argument('--optimizer', '-o', default='SGD', type=str,
                        help='Optimization Algorithm')
    parser.add_argument('--dataset', '-d',
                        default="../Assignment4_datasets/fakes_dataset", type=str,
                        help='explicit dataset_path, it should contains train, val and test folder, and each one of them should contain fake and real folders')
    parser.add_argument('--margin', default=0.2, type=float,
                        help='margin for triplet loss (default: 0.2)'
                        )
    return parser.parse_args()


def main():
    """Parse arguments and train model on dataset."""
    args = parse_args()

    # Model
    model_name = args.model
    model = load_model(model_name, args.embedding_dimension)

    # Data
    print(f'==> Preparing data: {args.dataset.replace("_", " ")}..')

    train_dataset = load_dataset(dataset_path=args.dataset,
                                 dataset_part='train', triplet=model_name=="Triplet")
    val_dataset = load_dataset(dataset_path=args.dataset, dataset_part='val', triplet=model_name=="Triplet")
    test_dataset = load_dataset(dataset_path=args.dataset, dataset_part='test', triplet=model_name=="Triplet")

    # Loss
    criterion = choose_criterion(model_name)

    # Build optimizer
    optimizers = {
        'SGD': lambda: optim.SGD(model.parameters(),
                                 lr=args.lr,
                                 momentum=args.momentum),
        'Adam': lambda: optim.Adam(model.parameters(), lr=args.lr),
    }

    optimizer_name = args.optimizer
    if optimizer_name not in optimizers:
        raise ValueError(f'Invalid Optimizer name: {optimizer_name}')

    print(f"Building optimizer {optimizer_name}...")
    optimizer = optimizers[args.optimizer]()
    print(optimizer)

    optimizer_params = optimizer.param_groups[0].copy()
    # remove the parameter values from the optimizer parameters for a cleaner
    # log
    del optimizer_params['params']

    # Batch size
    batch_size = args.batch_size

    # Training Logging Parameters
    logging_parameters = LoggingParameters(model_name=model_name,
                                           dataset_path=args.dataset,
                                           optimizer_name=optimizer_name,
                                           optimizer_params=optimizer_params,)

    # Create an abstract trainer to train the model with the data and parameters
    # above:
    trainer = Trainer(model=model,
                      optimizer=optimizer,
                      criterion=criterion,
                      batch_size=batch_size,
                      triplet=model_name=="Triplet",
                      train_dataset=train_dataset,
                      validation_dataset=val_dataset,
                      test_dataset=test_dataset,
                      margin=args.margin)

    # Train, evaluate and test the model:
    trainer.run(epochs=args.epochs, logging_parameters=logging_parameters)


if __name__ == '__main__':
    main()
