from .Errors import ImportNotFoundException, DataWrongException
from typing import Any, Literal

import numpy as np
import copy
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from .FileManagement import Export, create_directory

has_mps = torch.backends.mps.is_built()
device = "mps" if has_mps else "cuda" if torch.cuda.is_available() else "cpu"


class BoundaryDeviationLoss(nn.Module):
    def __init__(self):
        super(BoundaryDeviationLoss, self).__init__()

    def forward(self, inputs, targets) -> torch.Tensor:
        inputs_mean = torch.mean(inputs, 1, True)

        # Compute the lower and upper bounds for each row in a
        lower_bound = torch.min(targets, dim=1, keepdim=True).values
        upper_bound = torch.max(targets, dim=1, keepdim=True).values

        # Create masks for rows where a_pred_mean is within the bounds and where it's not
        within_bounds_mask = (inputs_mean >= lower_bound) & (inputs_mean <= upper_bound)
        outside_bounds_mask = ~within_bounds_mask

        # Compute the loss
        loss_within_bounds = torch.zeros_like(inputs_mean)
        loss_outside_bounds = torch.min(torch.abs(inputs_mean - lower_bound), torch.abs(inputs_mean - upper_bound))

        # Final loss
        loss: torch.Tensor = torch.where(within_bounds_mask, loss_within_bounds, loss_outside_bounds)
        return loss.mean()


class EarlyStopping:
    def __init__(self, patience=5, min_delta=0, restore_best_weights=True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_model = None
        self.best_loss = None
        self.counter = 0
        self.status = ""

    def __call__(self, model, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.best_model = copy.deepcopy(model.state_dict())
        elif self.best_loss - val_loss >= self.min_delta:
            self.best_model = copy.deepcopy(model.state_dict())
            self.best_loss = val_loss
            self.counter = 0
            self.status = f"Improvement found, counter reset to {self.counter}"
        else:
            self.counter += 1
            self.status = f"No improvement in the last {self.counter} epochs"
            if self.counter >= self.patience:
                self.status = f"Early stopping triggered after {self.counter} epochs."
                if self.restore_best_weights:
                    model.load_state_dict(self.best_model)
                return True
        return False


class PyTorchModel:
    def __init__(self,
                 hidden_layer_nodes: int,
                 out_nodes: int = 2,
                 epochs: int = 500,
                 batch_size: int = 38,
                 momentum: float = 0.2,
                 learning_rate: float = 0.001,
                 regularization: float = 0.1,
                 n_splits: int = 5,
                 shuffle: bool = True,
                 X: Any = None,
                 y: Any = None,
                 model: Literal['sequential', 'module', 'module_list'] = 'sequential',
                 with_dropout: bool = False,
                 r_in: float = 0.8,
                 r_h: float = 0.5
                 ) -> None:
        try:
            from sklearn.model_selection import KFold
        except Exception:
            raise ImportNotFoundException("Kfold cross-validation requires scikit-learn to be installed.")

        # dropout variables
        self.with_dropout = with_dropout
        self.R_IN = r_in
        self.R_H = r_h

        # checking training and testing data assignment
        if X is None or y is None:
            raise DataWrongException("X and y are required for this model to work")

        # checking model assignment
        if model == 'sequential':
            self.model = nn.Sequential()
        elif model == 'module':
            self.model = nn.Module()
        elif model == 'module_list':
            self.model = nn.ModuleList()
        else:
            raise DataWrongException(f"{model} is not a valid model")

        if device == "cuda":
            self.model = self.model.cuda()
        self.X = X
        self.y = y

        self.HIDDEN_LAYER_NODES = hidden_layer_nodes
        self.OUT_NODES = out_nodes

        self.EPOCHS = epochs
        self.BATCH_SIZE = batch_size

        self.MOMENTUM = momentum
        self.LEARNING_RATE = learning_rate
        self.REGULATION = regularization

        self.k_fold = KFold(n_splits=n_splits, shuffle=shuffle)

        self.fold_count = 0

        self.scores_mse = []
        self.scores_accuracy = []
        self.scores_CE = []

        self.oos_y = []
        self.oos_pred = []

        self.train_losses = []
        self.val_losses = []

        create_directory("./exports")
        self.exportFile = Export(
            f"./exports/{self.EPOCHS}_{self.BATCH_SIZE}_{self.HIDDEN_LAYER_NODES}_{self.LEARNING_RATE}_{self.MOMENTUM}.txt")

    def print_model(self) -> None:
        print(self.model)
        print(f"HIDDEN_LAYER_NODES={self.HIDDEN_LAYER_NODES}   OUT_NODES = {self.OUT_NODES}   EPOCHS={self.EPOCHS}   " +
              f"BATCH_SIZE={self.BATCH_SIZE}   LEARNING RATE={self.LEARNING_RATE}   MOMENTUM={self.MOMENTUM}")
        if self.with_dropout:
            print(f"WITH DROPOUT   R_IN: {self.R_IN}   R_H: {self.R_H}")

    def get_val_losses_train_losses(self):
        return self.val_losses, self.train_losses

    def append_layer(self, module_name: Literal['linear', 'relu'] = 'linear', in_nodes: int = None,
                     out_nodes: int = None) -> None:
        if module_name == 'linear' and device == "cuda":
            self.model.append(nn.Linear(in_features=in_nodes, out_features=out_nodes).cuda())
        elif module_name == 'linear':
            self.model.append(nn.Linear(in_features=in_nodes, out_features=out_nodes).cuda())
        elif module_name == 'relu' and device == "cuda":
            self.model.append(nn.ReLU().cuda())
        elif module_name == 'relu':
            self.model.append(nn.ReLU())
        else:
            DataWrongException(f"Module {module_name} is not a valid module")

    def create_default_model(self) -> None:
        self.append_layer(module_name='linear', in_nodes=np.shape(self.X)[1], out_nodes=self.HIDDEN_LAYER_NODES)
        self.append_layer(module_name='relu')
        self.append_layer(module_name='linear', in_nodes=self.HIDDEN_LAYER_NODES, out_nodes=self.OUT_NODES)

    def train_test(self, with_early_stopping: bool = False, patience: int = 5,
                   optimizer_name: Literal['adam', 'sgd'] = 'adam') -> None:
        # Convert to PyTorch Tensors
        self.exportFile.append_to_txt(f"MULTILAYER PERCEPTOR\n" +
                                      f"hidden_layer_nodes=(2 * INPUT_NODES) - (INPUT_NODES/8) = {self.HIDDEN_LAYER_NODES}, epochs={self.EPOCHS}, batch_size={self.BATCH_SIZE}" +
                                      f", learning rate={self.LEARNING_RATE}, momentum={self.MOMENTUM}\n" +
                                      f"{f'WITH DROPOUT   R_IN: {self.R_IN}, R_H: {self.R_H}'if self.with_dropout else 'NO DROPOUT'}\n" +
                                      f"{''.join('-' for x in range(100))}\n")
        x = torch.tensor(self.X, dtype=torch.float32, device=device)
        y = torch.tensor(self.y, dtype=torch.float32, device=device)

        # Set random seed for reproducibility
        torch.manual_seed(42)

        fold = 0
        for train_idx, test_idx in self.k_fold.split(x):
            fold += 1
            self.exportFile.append_to_txt(f"Fold #{fold}\n")
            print(f"Fold #{fold}")

            x_train, x_test = x[train_idx], x[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            # PyTorch DataLoader
            train_dataset = TensorDataset(x_train, y_train)
            train_loader = DataLoader(train_dataset, batch_size=self.BATCH_SIZE, shuffle=True)

            # initialize optimizer
            if optimizer_name == 'adam':
                optimizer = optim.Adam(self.model.parameters(), lr=self.LEARNING_RATE)
            elif optimizer_name == 'sgd':
                optimizer = optim.SGD(self.model.parameters(), lr=self.LEARNING_RATE, momentum=self.MOMENTUM)
            else:
                raise DataWrongException(f"{optimizer_name} is not a valid optimizer")

            loss_fn = nn.MSELoss()

            # Early Stopping variables
            if with_early_stopping:
                best_loss = float('inf')
                early_stopping_counter = 0
                es = EarlyStopping(patience=patience)

            # Training loop
            epoch = 0
            done = False

            while not done and epoch < self.EPOCHS:
                epoch += 1
                train_loss_epoch = 0.0
                num_loss = 0.0
                num_batches = 0

                self.model.train()
                for x_batch, y_batch in train_loader:
                    optimizer.zero_grad()
                    output = self.model(x_batch)
                    loss = loss_fn(output, y_batch)
                    loss.backward()
                    optimizer.step()

                    train_loss_epoch += loss.item()
                    num_batches += 1

                train_loss_epoch /= num_batches
                self.train_losses.append(train_loss_epoch)

                # Validation
                self.model.eval()
                with torch.no_grad():
                    val_output = self.model(x_test)
                    val_loss = loss_fn(val_output, y_test)

                val_loss_epoch = val_loss.item()
                self.val_losses.append(val_loss_epoch)
                if with_early_stopping:
                    if es(self.model, val_loss) and with_early_stopping:
                        done = True

            if with_early_stopping:
                print(f"Epoch {epoch}/{self.EPOCHS}, Validation Loss: "
                      f"{val_loss.item()}, {es.status}")
            else:
                print(f"Epoch {epoch}/{self.EPOCHS}, Validation Loss: {val_loss.item()}")
                self.exportFile.append_to_txt(f"Epoch {epoch}/{self.EPOCHS}, Validation Loss: {val_loss.item()}\n")

        # Final evaluation
        self.model.eval()
        with torch.no_grad():
            oos_pred = self.model(x_test)
        score = torch.sqrt(loss_fn(oos_pred, y_test)).item()
        self.exportFile.append_to_txt(f"Fold score (RMSE): {score}\n")
        print(f"Fold score (RMSE): {score}")
        self.plot_losses()

    def plot_losses(self):
        plt.plot(range(1, len(self.train_losses) + 1), self.train_losses, label='Training Loss')
        plt.plot(range(1, len(self.val_losses) + 1), self.val_losses, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss Curves\n' +
                  f'Epoch: {self.EPOCHS} Batch size: {self.BATCH_SIZE} Hidden Layer Nodes: {self.HIDDEN_LAYER_NODES}\n' +
                  f'Learning Rate: {self.LEARNING_RATE}   Momentum={self.MOMENTUM}')
        plt.legend()
        plt.savefig('foo.png', bbox_inches='tight')
        plt.show()
