import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# to make sure that the results are reproducible
torch.manual_seed(42)
np.random.seed(42)

# define the dataset class
class AFSDataset(Dataset):
    def __init__(self, afs_file, theta_file):
        self.afs_data = np.loadtxt(afs_file)
        
        self.scaler = StandardScaler()
        self.afs_data = self.scaler.fit_transform(self.afs_data)
        
        if theta_file:
            self.theta_values = np.loadtxt(theta_file)
            self.has_labels = True
        else:
            self.has_labels = False
    
    def __len__(self):
        return len(self.afs_data)
    
    def __getitem__(self, idx):
        afs = torch.tensor(self.afs_data[idx], dtype=torch.float32)
        
        if self.has_labels:
            theta = torch.tensor(self.theta_values[idx], dtype=torch.float32)
            return afs, theta
        else:
            return afs

# define the feedforward neural network model
class FeedForwardNN(nn.Module):
    def __init__(self, input_size):
        super(FeedForwardNN, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
    
    def forward(self, x):
        return self.model(x).squeeze()

# use the train dataset and validation dataset to train the model
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=50):
    train_losses = []
    val_losses = []
    
    for epoch in range(num_epochs):
        # training phase
        model.train()
        running_loss = 0.0
        
        for inputs, targets in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
        
        epoch_train_loss = running_loss / len(train_loader.dataset)
        train_losses.append(epoch_train_loss)
        
        # validation phase
        model.eval()
        running_val_loss = 0.0
        
        with torch.no_grad():
            for inputs, targets in val_loader:
                outputs = model(inputs)
                val_loss = criterion(outputs, targets)
                running_val_loss += val_loss.item() * inputs.size(0)
        
        epoch_val_loss = running_val_loss / len(val_loader.dataset)
        val_losses.append(epoch_val_loss)
        
        # print the loss every 5 epochs
        if (epoch + 1) % 5 == 0:
            print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {epoch_train_loss:.4f}, Val Loss: {epoch_val_loss:.4f}')
    
    return train_losses, val_losses

def evaluate_model(model, test_loader):
    model.eval()
    all_predictions = []
    all_actual = []
    
    with torch.no_grad():
        for inputs, targets in test_loader:
            outputs = model(inputs)
            
            if outputs.ndim == 0:
                outputs = outputs.unsqueeze(0)
            
            if targets.ndim == 0:
                targets = targets.unsqueeze(0)
                
            batch_predictions = outputs.cpu().numpy()
            batch_targets = targets.cpu().numpy()
            
            if len(batch_predictions) != len(batch_targets):
                print(f"The batch size is not matching! Predictions: {len(batch_predictions)}, Targets: {len(batch_targets)}")

                min_len = min(len(batch_predictions), len(batch_targets))
                batch_predictions = batch_predictions[:min_len]
                batch_targets = batch_targets[:min_len]
            
            all_predictions.extend(batch_predictions)
            all_actual.extend(batch_targets)
    
    predictions = np.array(all_predictions)
    actual = np.array(all_actual)
    
    # check if the final arrays have the same length
    if len(predictions) != len(actual):
        print(f"Error: Final arrays have different lengths! Predictions: {len(predictions)}, Actual: {len(actual)}")

        min_len = min(len(predictions), len(actual))
        predictions = predictions[:min_len]
        actual = actual[:min_len]
    
    mse = mean_squared_error(actual, predictions)
    rmse = np.sqrt(mse)
    r2 = r2_score(actual, predictions)
    correlation = np.corrcoef(predictions, actual)[0, 1]
    
    return {
        'mse': mse,
        'rmse': rmse,
        'r2': r2,
        'correlation': correlation,
        'predictions': predictions,
        'actual': actual
    }

def plot_results_val(train_losses, val_losses, predictions, actual):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # plot the loss curves
    epochs = range(1, len(train_losses) + 1)
    ax1.plot(epochs, train_losses, 'b', label='Training Loss')
    ax1.plot(epochs, val_losses, 'r', label='Validation Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    ax1.legend()
    
    # plot the scatter plot of predicted vs actual theta values
    ax2.scatter(actual, predictions, alpha=0.5)
    min_val = min(min(actual), min(predictions))
    max_val = max(max(actual), max(predictions))
    ax2.plot([min_val, max_val], [min_val, max_val], 'r--')
    ax2.set_title('Predicted vs Actual Theta Values')
    ax2.set_xlabel('Actual')
    ax2.set_ylabel('Predicted')
    
    plt.tight_layout()
    plt.savefig('afs_valid_theta_results.png')
    plt.show()

def plot_results_test(train_losses, test_losses, predictions, actual):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # plot the loss curves
    epochs = range(1, len(train_losses) + 1)
    ax1.plot(epochs, train_losses, 'b', label='Training Loss')
    ax1.plot(epochs, test_losses, 'r', label='Test Loss')
    ax1.set_title('Training and Test Loss')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    ax1.legend()
    
    # plot the scatter plot of predicted vs actual theta values
    ax2.scatter(actual, predictions, alpha=0.5)
    min_val = min(min(actual), min(predictions))
    max_val = max(max(actual), max(predictions))
    ax2.plot([min_val, max_val], [min_val, max_val], 'r--')
    ax2.set_title('Predicted vs Actual Theta Values')
    ax2.set_xlabel('Actual')
    ax2.set_ylabel('Predicted')
    
    plt.tight_layout()
    plt.savefig('afs_test_theta_results.png')
    plt.show()


def main():
    train_afs_file = 'AFS_train.txt'
    train_theta_file = 'theta_train.txt'
    test_afs_file = 'AFS_test.txt'
    test_theta_file = 'theta_test.txt'
    
    train_dataset = AFSDataset(train_afs_file, train_theta_file)
    
    # setting the train and validation dataset
    train_size = int(0.8 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [train_size, val_size])
    
    # load the train and validation dataset
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32)
    
    # get the input size
    sample_afs, _ = train_dataset[0]
    input_size = sample_afs.shape[0]
    
    # create the model
    model = FeedForwardNN(input_size)
    
    # define the loss function and the optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # begin training the model
    print("Begin to train the model...")
    train_losses, val_losses = train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=50)
    print("Model training completed!")
    

    print("Evaluating the model on the validation set...")
    eval_results = evaluate_model(model, val_loader)
    
    print(f"Mean Square Error (MSE): {eval_results['mse']:.4f}")
    print(f"Root Mean Square Error (RMSE): {eval_results['rmse']:.4f}")
    print(f"Determination Coefficient (R²): {eval_results['r2']:.4f}")
    print(f"Correlation: {eval_results['correlation']:.4f}")
    
    plot_results_val(train_losses, val_losses, eval_results['predictions'], eval_results['actual'])
    
    if test_theta_file:
        print("\nTesting the model on the test dataset...")
        test_dataset = AFSDataset(test_afs_file, test_theta_file)
        test_loader = DataLoader(test_dataset, batch_size=32)
        test_results = evaluate_model(model, test_loader)
        
        print(f"Test dataset MSE: {test_results['mse']:.4f}")
        print(f"Test dataset RMSE: {test_results['rmse']:.4f}")
        print(f"Test dataset R²: {test_results['r2']:.4f}")
        print(f"Test correlation: {test_results['correlation']:.4f}")
        plot_results_test(train_losses, val_losses, test_results['predictions'], test_results['actual'])

    
    else:
        print("\n predicting the theta values for the test dataset... if no test_theta_file is provided")
        test_dataset = AFSDataset(test_afs_file)
        batch_size = 32
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
        model.eval()
        predictions = []
        
        
        with torch.no_grad():
            for inputs in test_loader:
                outputs = model(inputs)
                predictions.extend(outputs.numpy())
        
        np.savetxt('predicted_theta_values.txt', predictions)
        print("The prediction result is saved to'predicted_theta_values.txt'")

if __name__ == "__main__":
    main()