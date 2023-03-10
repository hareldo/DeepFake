"""Train models on a given dataset."""
import os
import json
from dataclasses import dataclass
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Function
from torch.nn.modules.distance import PairwiseDistance
from common import OUTPUT_DIR, CHECKPOINT_DIR
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class TripletLoss(Function):

    def __init__(self, margin):
        super(TripletLoss, self).__init__()
        self.margin = margin
        self.pdist = PairwiseDistance(p=2)

    def forward(self, anchor, positive, negative):
        pos_dist = self.pdist.forward(anchor, positive)
        neg_dist = self.pdist.forward(anchor, negative)

        hinge_dist = torch.clamp(self.margin + pos_dist - neg_dist, min=0.0)
        loss = torch.mean(hinge_dist)

        return loss


@dataclass
class LoggingParameters:
    """Data class holding parameters for logging."""
    model_name: str
    dataset_path: str
    optimizer_name: str
    optimizer_params: dict

# pylint: disable=R0902, R0913, R0914
class Trainer:
    """Abstract model trainer on a binary classification task."""
    def __init__(self,
                 model: nn.Module,
                 optimizer: torch.optim,
                 criterion,
                 batch_size: int,
                 triplet: bool,
                 train_dataset: Dataset,
                 validation_dataset: Dataset,
                 test_dataset: Dataset,
                 margin: int):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.criterion = criterion
        self.batch_size = batch_size
        self.triplet = triplet
        self.train_dataset = train_dataset
        self.validation_dataset = validation_dataset
        self.test_dataset = test_dataset
        self.epoch = 0
        self.margin = margin

    def train_one_epoch(self):
        """Train the model for a single epoch on the training dataset.
        Returns:
            (avg_loss, accuracy): tuple containing the average loss and
            accuracy across all dataset samples.
        """
        self.model.train()
        total_loss = 0
        avg_loss = 0
        accuracy = 0
        nof_samples = 0
        correct_labeled_samples = 0

        train_dataloader = DataLoader(self.train_dataset,
                                      self.batch_size,
                                      shuffle=True)
        print_every = int(len(train_dataloader) / 10)

        for batch_idx, (inputs, targets) in enumerate(train_dataloader):
            # training steps
            self.optimizer.zero_grad()

            if self.triplet:
                anchor, pos, neg = inputs[0].to(device), inputs[1].to(device), inputs[2].to(device)
                anc_embeddings = self.model(anchor)
                pos_embeddings = self.model(pos)
                neg_embeddings = self.model(neg)

                crit = PairwiseDistance(p=2)
                pos_dists = crit(anc_embeddings, pos_embeddings)
                neg_dists = crit(anc_embeddings, neg_embeddings)

                all = (neg_dists - pos_dists < self.margin).cpu().numpy().flatten()
                valid_triplets = np.where(all == 1)

                anc_valid_embeddings = anc_embeddings[valid_triplets]
                pos_valid_embeddings = pos_embeddings[valid_triplets]
                neg_valid_embeddings = neg_embeddings[valid_triplets]

                loss = self.criterion(anc_valid_embeddings, pos_valid_embeddings, neg_valid_embeddings)
                correct_labeled_samples += torch.sum(pos_dists[valid_triplets] < neg_dists[valid_triplets])
                nof_samples += len(valid_triplets[0])
            else:
                inputs, targets = inputs.to(device), targets.to(device)

                pred_probability = self.model(inputs)   # prediction probability is size 2 (real, fake)
                loss = self.criterion(pred_probability, targets)

                pred = torch.argmax(pred_probability, dim=1)
                correct_labeled_samples += torch.sum(pred == targets)
                nof_samples += len(targets)

            loss.backward()
            self.optimizer.step()
            total_loss += loss
            avg_loss = total_loss.item() / (batch_idx + 1)

            # recording step
            if nof_samples > 0:
                accuracy = correct_labeled_samples.item() / nof_samples * 100
            else:
                accuracy = 100
            if batch_idx % print_every == 0 or \
                    batch_idx == len(train_dataloader) - 1:
                print(f'Epoch [{self.epoch:03d}] | Loss: {avg_loss:.3f} | '
                      f'Acc: {accuracy:.2f}[%] '
                      f'({correct_labeled_samples}/{nof_samples})')

        return avg_loss, accuracy

    def evaluate_model_on_dataloader(self, dataset: torch.utils.data.Dataset, debug=False):
        """Evaluate model loss and accuracy for dataset.

        Args:
            dataset: the dataset to evaluate the model on.
            debug: Debug mode

        Returns:
            (avg_loss, accuracy): tuple containing the average loss and
            accuracy across all dataset samples.
        """
        self.model.eval()
        dataloader = DataLoader(dataset,
                                batch_size=self.batch_size,
                                shuffle=False)
        total_loss = 0
        avg_loss = 0
        accuracy = 0
        nof_samples = 0
        correct_labeled_samples = 0
        print_every = max(int(len(dataloader) / 10), 1)

        for batch_idx, (inputs, targets) in enumerate(dataloader):
            with torch.no_grad():

                if self.triplet:
                    anchor, pos, neg = inputs[0].to(device), inputs[1].to(device), inputs[2].to(device)

                    anc_embeddings = self.model(anchor)
                    pos_embeddings = self.model(pos)
                    neg_embeddings = self.model(neg)

                    # Split the embeddings into Anchor, Positive, and Negative embeddings
                    batch_length = anchor.shape[0]
                    # anc_embeddings = embeddings[:batch_length]
                    # pos_embeddings = embeddings[batch_length: batch_length * 2]
                    # neg_embeddings = embeddings[batch_length * 2:]

                    crit = PairwiseDistance(p=2)
                    pos_dists = crit(anc_embeddings, pos_embeddings)
                    neg_dists = crit(anc_embeddings, neg_embeddings)
                    # pos_dists = self.criterion(anc_embeddings, pos_embeddings)
                    # neg_dists = self.criterion(anc_embeddings, neg_embeddings)

                    all = (neg_dists - pos_dists < self.margin).cpu().numpy().flatten()
                    valid_triplets = np.where(all == 1)

                    anc_valid_embeddings = anc_embeddings[valid_triplets]
                    pos_valid_embeddings = pos_embeddings[valid_triplets]
                    neg_valid_embeddings = neg_embeddings[valid_triplets]

                    # Calculate triplet loss
                    # loss = TripletLoss(margin=self.margin).forward(
                    #     anchor=anc_embeddings,
                    #     positive=pos_embeddings,
                    #     negative=neg_embeddings
                    # )

                    loss = self.criterion(anc_valid_embeddings, pos_valid_embeddings, neg_valid_embeddings)
                    # loss = self.criterion(anc_embeddings, pos_embeddings, neg_embeddings)
                    correct_labeled_samples += torch.sum(pos_dists[valid_triplets] < neg_dists[valid_triplets])
                    # correct_labeled_samples += torch.sum(pos_valid_embeddings < neg_valid_embeddings)

                else:
                    inputs, targets = inputs.to(device), targets.to(device)
                    pred_probability = self.model(inputs)
                    loss = self.criterion(pred_probability, targets)
                    prediction = torch.argmax(pred_probability, dim=1)
                    correct_labeled_samples += torch.sum(prediction == targets)

                total_loss += loss
                avg_loss = total_loss.item() / (batch_idx + 1)

                # Analyze
                nof_samples += len(targets)
                accuracy = (correct_labeled_samples.item() / nof_samples) * 100

            if batch_idx % print_every == 0 or batch_idx == len(dataloader) - 1:
                print(f'Epoch [{self.epoch:03d}] | Loss: {avg_loss:.3f} | '
                      f'Acc: {accuracy:.2f}[%] '
                      f'({correct_labeled_samples}/{nof_samples})')
        if debug:
            pca = PCA(n_components=2)
            pos_2d = pca.fit_transform(pos_valid_embeddings.cpu())
            neg_2d = pca.fit_transform(neg_valid_embeddings.cpu())
            plt.figure()
            plt.scatter(pos_2d[:, 0], pos_2d[:, 1])
            plt.scatter(neg_2d[:, 0], neg_2d[:, 1])
            plt.show()

        return avg_loss, accuracy

    def validate(self):
        """Evaluate the model performance."""
        return self.evaluate_model_on_dataloader(self.validation_dataset)

    def test(self):
        """Test the model performance."""
        return self.evaluate_model_on_dataloader(self.test_dataset)

    @staticmethod
    def write_output(logging_parameters: LoggingParameters, data: dict):
        """Write logs to json.

        Args:
            logging_parameters: LoggingParameters. Some parameters to log.
            data: dict. Holding a dictionary to dump to the output json.
        """
        if not os.path.isdir(OUTPUT_DIR):
            os.makedirs(OUTPUT_DIR)

        output_filename = f"{logging_parameters.model_name}_" \
                          f"{logging_parameters.optimizer_name}.json"
        output_filepath = os.path.join(OUTPUT_DIR, output_filename)

        print(f"Writing output to {output_filepath}")
        # Load output file
        if os.path.exists(output_filepath):
            # pylint: disable=C0103
            with open(output_filepath, 'r', encoding='utf-8') as f:
                all_output_data = json.load(f)
        else:
            all_output_data = []

        # Add new data and write to file
        all_output_data.append(data)
        # pylint: disable=C0103
        with open(output_filepath, 'w', encoding='utf-8') as f:
            json.dump(all_output_data, f, indent=4)

    def run(self, epochs, logging_parameters: LoggingParameters):
        """Train, evaluate and test model on dataset, finally log results."""
        output_data = {
            "model": logging_parameters.model_name,
            "dataset": logging_parameters.dataset_path,
            "optimizer": {
                "name": logging_parameters.optimizer_name,
                "params": logging_parameters.optimizer_params,
            },
            "train_loss": [],
            "train_acc": [],
            "val_loss": [],
            "val_acc": [],
            "test_loss": [],
            "test_acc": [],
        }
        best_acc = 0
        model_filename = f"{logging_parameters.model_name}_" \
                         f"{logging_parameters.optimizer_name}.pt"
        checkpoint_filename = os.path.join(CHECKPOINT_DIR, model_filename)
        for self.epoch in range(1, epochs + 1):
            print(f'Epoch {self.epoch}/{epochs}')

            train_loss, train_acc = self.train_one_epoch()
            val_loss, val_acc = self.validate()
            test_loss, test_acc = self.test()

            output_data["train_loss"].append(train_loss)
            output_data["train_acc"].append(train_acc)
            output_data["val_loss"].append(val_loss)
            output_data["val_acc"].append(val_acc)
            output_data["test_loss"].append(test_loss)
            output_data["test_acc"].append(test_acc)

            # Save checkpoint
            if val_acc > best_acc:
                print(f'Saving checkpoint {checkpoint_filename}')
                state = {
                    'model': self.model.state_dict(),
                    'val_acc': val_acc,
                    'test_acc': test_acc,
                    'epoch': self.epoch,
                }
                torch.save(state, checkpoint_filename)
                best_acc = val_acc
        self.write_output(logging_parameters, output_data)
