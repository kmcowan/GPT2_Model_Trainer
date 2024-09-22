import matplotlib.pyplot as plt
import numpy as np


class RealTimeLossMonitor:
    def __init__(self, patience=5):
        """
        Initialize the RealTimeLossMonitor for tracking training and validation loss.

        Parameters:
        - patience (int): Number of epochs to wait for validation loss improvement before early stopping.
        """
        self.training_losses = []
        self.validation_losses = []
        self.patience = patience

        # Initialize phase tracking
        self.initial_phase_end = None
        self.middle_phase_end = None
        self.plateau_start = None
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        self.optimal_epoch = None

    def add_epoch_loss(self, train_loss, val_loss):
        """
        Add a single epoch's training and validation loss to the monitor.

        Parameters:
        - train_loss (float): Training loss for the current epoch.
        - val_loss (float): Validation loss for the current epoch.
        """
        self.training_losses.append(train_loss)
        self.validation_losses.append(val_loss)

        # Check for phase transitions
        epoch = len(self.training_losses)
        if epoch > 1:
            train_loss_diff = self.training_losses[-2] - self.training_losses[-1]
            if self.initial_phase_end is None and train_loss_diff < 0.01:
                self.initial_phase_end = epoch
                print(f"Transition to Middle Phase detected at epoch {epoch}")

            if self.initial_phase_end is not None and self.middle_phase_end is None and train_loss_diff < 0.001:
                self.middle_phase_end = epoch
                self.plateau_start = epoch + 1
                print(f"Transition to Plateau Phase detected at epoch {epoch}")

        # Check for validation loss improvement
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            self.optimal_epoch = epoch
            self.patience_counter = 0  # Reset patience counter when improvement is found
        else:
            self.patience_counter += 1

        # Check for early stopping condition
        if self.patience_counter >= self.patience:
            print(
                f"Early stopping triggered at epoch {epoch}. No improvement in validation loss for {self.patience} epochs.")
            return False  # Return False to indicate training should stop

        return True  # Return True to indicate training can continue

    def get_summary(self):
        """
        Provides a summary of the training process.

        Returns:
        - dict: A dictionary containing the training phases and optimal epoch information.
        """
        return {
            "initial_phase_end": self.initial_phase_end,
            "middle_phase_end": self.middle_phase_end,
            "plateau_start": self.plateau_start,
            "optimal_epoch": self.optimal_epoch,
            "best_val_loss": self.best_val_loss
        }

    def plot_losses(self):
        """
        Plots the training and validation loss over the epochs.
        """
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, len(self.training_losses) + 1), self.training_losses, label='Training Loss', color='blue')
        plt.plot(range(1, len(self.validation_losses) + 1), self.validation_losses, label='Validation Loss',
                 color='orange')

        if self.optimal_epoch:
            plt.axvline(x=self.optimal_epoch, linestyle='--', color='red', label=f'Optimal Epoch: {self.optimal_epoch}')

        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss Monitoring')
        plt.legend()
        plt.grid(True)
        plt.show()
