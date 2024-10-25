from copy import deepcopy
import tqdm
import os
import datetime
from pathlib import Path
import shutil

import numpy as np

import torch
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import torch.nn as nn
from torchmetrics import Accuracy, Recall, Precision, F1Score

'''
SKLearn Compatible Implementation of DeepDrebin
'''


class DrebinDNN(nn.Module):
    """
    Network architecture used by Grosse et al. in the paper
    'Adversarial Examples for Malware Detection'
    """

    def __init__(self, input_size):
        super(DrebinDNN, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, 200),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(200, 200),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(200, 2),
        )

    def forward(self, x):
        return self.layers(x)


class DeepDrebin:
    def __init__(self):
        # use CUDA if available
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # hyper parameters
        self.num_epochs = 10
        self.batch_size = 64
        self.learning_rate = 0.05
        self.split_ratio = 0.66

        # number fo validation months of data
        self.train_months = 12
        self.num_val_month = 3

        self.random_seed = 0x10c0ffee
        self.num_samples = None  # don't use all samples of the dataset (None if all samples should be used)

        # file paths
        self.project_dir = Path(__file__).resolve().parents[0]
        self.model_dir = os.path.join(self.project_dir, 'data/results/deepdrebin/models')
        self.result_dir = os.path.join(self.project_dir, 'data/results/deepdrebin/')

        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)

        self.last_model_path = os.path.join(self.model_dir, 'last_model-{}.pth')
        self.best_model_path = os.path.join(self.model_dir, 'best_model-{}.pth')

        timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        tensorboard_path = "logs/tensorboard/" + timestamp

        self.checkpoint = True
        self.desc = None

        self.model = None
        self.optimizer = None
        self.criterion = None

        self.train_time = None
        self.t = None

        self.pois_idx = None  # The list of indexes of poisoning data in the training set

    def setup(self, desc, train_time, X, y, t, pois_idx=None):
        self.desc = desc
        self.train_time = train_time
        self.t = t

        if pois_idx:
            self.pois_idx = pois_idx
        else:
            self.pois_idx = []

        # Convert data to tensor
        y = np.array(y)
        X = X.tocoo()
        X_tensor_tmp = torch.sparse_coo_tensor(indices=torch.LongTensor([X.row.tolist(), X.col.tolist()]),
                                               values=torch.LongTensor(X.data.astype(np.int32)),
                                               size=X.shape)

        # Convert the csr_matrix to numpy array, then to tensor
        X_tensor = X_tensor_tmp.to_dense().type(torch.FloatTensor)
        y_tensor = torch.from_numpy(y.astype(np.int_))

        dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor)

        # Initialize Model
        features, labels = dataset[0]
        n_features = features.shape[0]
        print('feature size is {}'.format(n_features))
        self.model = DrebinDNN(n_features)
        self.model.to(device=self.device)

        # Define Loss and Optimizer
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(params=self.model.parameters(), lr=self.learning_rate)

    def fit(self, X_train, y_train):

        # Convert data to tensor
        y = np.array(y_train)
        X = X_train.tocoo()
        X_tensor_tmp = torch.sparse_coo_tensor(indices=torch.LongTensor([X.row.tolist(), X.col.tolist()]),
                                               values=torch.LongTensor(X.data.astype(np.int32)),
                                               size=X.shape)

        # Convert the csr_matrix to numpy array, then to tensor
        X_tensor = X_tensor_tmp.to_dense().type(torch.FloatTensor)
        y_tensor = torch.from_numpy(y.astype(np.int_))

        dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor)

        # Prepare training Dataset
        indices = [x for x in range(X_train.shape[0])]
        np.random.shuffle(indices)
        train_size = int(self.split_ratio * len(indices))

        train_indices = indices[:train_size]
        val_indices = indices[train_size:]

        train_sampler = SubsetRandomSampler(train_indices)
        val_sampler = SubsetRandomSampler(val_indices)

        train_loader = DataLoader(dataset=dataset, batch_size=self.batch_size, sampler=train_sampler)
        val_loader = DataLoader(dataset=dataset, batch_size=self.batch_size, sampler=val_sampler)

        self.optimizer = torch.optim.SGD(params=self.model.parameters(), lr=self.learning_rate)

        # Train Model
        accuracy = Accuracy(num_classes=2, average='macro', task='binary').to(device=self.device)
        f1score = F1Score(num_classes=2, average='macro', task='binary').to(device=self.device)
        precision = Precision(num_classes=2, average='macro', task='binary').to(device=self.device)
        recall = Recall(num_classes=2, average='macro', task='binary').to(device=self.device)
        valid_max_f1score = 0.0
        print("Start Training")

        for epoch in range(self.num_epochs):
            ####################
            # Train Model
            ####################
            self.model.train()
            train_loss = 0.0

            for i, (inputs, labels) in enumerate(tqdm.tqdm(train_loader, desc='train')):
                # Push data to device
                inputs = inputs.to(device=self.device)
                labels = labels.to(device=self.device)

                # forward step
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)

                # backward step
                self.optimizer.zero_grad()
                loss.backward()

                # update weights
                self.optimizer.step()

                train_loss += loss.item()

                _, predictions = torch.max(outputs.data, 1)

                # compute metrics on current batch
                accuracy(predictions, labels)
                f1score(predictions, labels)
                precision(predictions, labels)
                recall(predictions, labels)

            # reset metrics
            accuracy.reset()
            f1score.reset()
            precision.reset()
            recall.reset()

            ####################
            # Validate Model
            ####################
            self.model.eval()
            with torch.no_grad():
                for inputs, labels in tqdm.tqdm(val_loader, desc='val'):
                    # Push data to device
                    inputs = inputs.to(device=self.device)
                    labels = labels.to(device=self.device)

                    # Make predictions
                    outputs = self.model(inputs)
                    _, predictions = torch.max(outputs, 1)

                    # Calculate metrics
                    accuracy(predictions, labels)
                    f1score(predictions, labels)
                    recall(predictions, labels)
                    precision(predictions, labels)

                checkpoint = {
                    'epoch': epoch + 1,
                    'accuracy': accuracy.compute(),
                    'f1score': f1score.compute(),
                    'precision': precision.compute(),
                    'recall': recall.compute(),
                    'state_dict': self.model.state_dict(),
                    'optimizer': self.optimizer.state_dict(),
                }

                # save mode
                is_best_model = False
                if checkpoint['f1score'] > valid_max_f1score or valid_max_f1score == 0.0:
                    is_best_model = True
                    valid_max_f1score = checkpoint['f1score']

                self.save_checkpoint(is_best_model, checkpoint)

    def predict(self, X_test):

        # Convert the csr_matrix to numpy array to tensor to dataloader
        X = X_test.tocoo()
        X_tensor_tmp = torch.sparse_coo_tensor(indices=torch.LongTensor([X.row.tolist(), X.col.tolist()]),
                                               values=torch.LongTensor(X.data.astype(np.int32)),
                                               size=X.shape)
        X_tensor = X_tensor_tmp.to_dense().type(torch.FloatTensor)

        dummy_tensor = torch.tensor([i for i in range(X.shape[0])], dtype=torch.int)
        dataset = torch.utils.data.TensorDataset(X_tensor, dummy_tensor)

        test_loader = DataLoader(dataset=dataset, batch_size=self.batch_size, shuffle=False)

        # Load Best Model Weights
        _, _, _ = self.load_checkpoint(self.best_model_path.format(self.desc))

        # 2. Test Model
        self.model.eval()
        y_pred = []
        with torch.no_grad():
            for inputs, _ in test_loader:
                inputs = inputs.to(device=self.device)
                outputs = self.model(inputs)
                _, prediction = torch.max(outputs, 1)
                y_pred.extend(prediction.tolist())

        return y_pred

    def predict_proba(self, X_test):

        # Convert the csr_matrix to numpy array to tensor to dataloader
        X = X_test.tocoo()
        X_tensor_tmp = torch.sparse_coo_tensor(indices=torch.LongTensor([X.row.tolist(), X.col.tolist()]),
                                               values=torch.LongTensor(X.data.astype(np.int32)),
                                               size=X.shape)
        X_tensor = X_tensor_tmp.to_dense().type(torch.FloatTensor)
        dataset = torch.utils.data.TensorDataset(X_tensor)
        test_loader = DataLoader(dataset=dataset, batch_size=self.batch_size, shuffle=False)

        # Load Best Model Weights
        _, _, _ = self.load_checkpoint(self.best_model_path.format(self.desc))

        # 2. Test Model
        self.model.eval()
        y_pred_proba = []
        with torch.no_grad():
            for inputs, in test_loader:
                inputs = inputs.to(device=self.device)
                outputs = self.model(inputs)
                probabilities = torch.softmax(outputs, dim=1)
                y_pred_proba.extend(probabilities.tolist())

        return y_pred_proba

    def save_checkpoint(self, is_best, state):
        """
        is_best: is this the best checkpoint; min validation loss
        state: the check point to save
        """

        f_path = self.last_model_path.format(self.desc)
        # save checkpoint data to the path given, checkpoint_path
        torch.save(state, f_path)
        # if it is the best model, min validation loss

        if is_best:
            best_fpath = self.best_model_path.format(self.desc)
            # copy that checkpoint file the best path given, best_model_path
            shutil.copyfile(f_path, best_fpath)

    def load_checkpoint(self, checkpoint_fpath):
        """
        Load a model and (optionally) an optimizer from a checkpoint.

        Args:
        - checkpoint_fpath (str): Path to the saved checkpoint.
        - model (torch.nn.Module): Model to load the checkpoint into.
        - optimizer (torch.optim.Optimizer, optional): Optimizer to load from the checkpoint. Default is None.

        Returns:
        - model, optimizer (if provided), epoch value
        """
        # Check for the checkpoint file's existence
        if not os.path.exists(checkpoint_fpath):
            raise FileNotFoundError(f"No checkpoint found at '{checkpoint_fpath}'!")

        # Load the checkpoint
        checkpoint = torch.load(checkpoint_fpath,
                                map_location=lambda storage, loc: storage)  # Ensure the loading is device-agnostic

        # Validate the checkpoint keys
        if 'state_dict' not in checkpoint:
            raise KeyError("No state_dict found in checkpoint file!")
        if self.optimizer and 'optimizer' not in checkpoint:
            raise KeyError("No optimizer state found in checkpoint file!")

        # Load model state
        self.model.load_state_dict(checkpoint['state_dict'])

        # If an optimizer is provided, and its state is saved, load it
        if self.optimizer:
            self.optimizer.load_state_dict(checkpoint['optimizer'])

        # Get epoch number if it's available, else return None
        epoch = checkpoint.get('epoch', None)

        return self.model, self.optimizer, epoch
