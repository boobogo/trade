import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Generate data
x = np.linspace(1, 10, 1000)  # 1000 points between 1 and 10
y = np.log(x)  # log of x
print(x)
# Convert to PyTorch tensors
inputs = torch.tensor(x, dtype=torch.float32).unsqueeze(1)  # Shape (1000, 1)
targets = torch.tensor(y, dtype=torch.float32).unsqueeze(1)  # Shape (1000, 1)

# Split data into training, validation, and test sets
train_inputs, temp_inputs, train_targets, temp_targets = train_test_split(inputs, targets, test_size=0.3, random_state=42)
val_inputs, test_inputs, val_targets, test_targets = train_test_split(temp_inputs, temp_targets, test_size=0.5, random_state=42)

# Define the neural network model
class LogNN(nn.Module):
    def __init__(self, input_size, hidden1_size, hidden2_size, output_size):
        super(LogNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden1_size)  # First fully connected layer
        self.relu = nn.ReLU()  # Activation function
        self.fc2 = nn.Linear(hidden1_size, hidden2_size)  # Second fully connected layer
        self.fc3 = nn.Linear(hidden2_size, output_size)  # Third fully connected layer

    def forward(self, x):
        out = self.fc1(x)  # Linear transformation
        out = self.relu(out)  # Apply ReLU activation
        out = self.fc2(out)  # Linear transformation
        out = self.relu(out)  # Apply ReLU activation
        out = self.fc3(out)
        return out

# Hyperparameters
input_size = 1  # One input feature (x value)
hidden1_size = 64  # Number of neurons in the hidden1 layer
hidden2_size = 64  # Number of neurons in the hidden2 layer
output_size = 1  # One output feature (y value)
learning_rate = 0.01
num_epochs = 500

# Create the model, define the loss function and the optimizer
model = LogNN(input_size, hidden1_size, hidden2_size, output_size)
criterion = nn.MSELoss()  # Mean Squared Error Loss
optimizer = optim.Adam(model.parameters(), lr=learning_rate)  # Adam optimizer

# Training loop
train_losses = []
val_losses = []

for epoch in range(num_epochs):
    # Forward pass
    model.train()
    outputs = model(train_inputs)
    loss = criterion(outputs, train_targets)  # Compute the training loss

    # Backward pass and optimization
    optimizer.zero_grad()  # Clear the gradients
    loss.backward()  # Compute gradients
    optimizer.step()  # Update weights

    # Compute validation loss
    model.eval()
    with torch.no_grad():
        val_outputs = model(val_inputs)
        val_loss = criterion(val_outputs, val_targets)

    train_losses.append(loss.item())
    val_losses.append(val_loss.item())

    if (epoch+1) % 50 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.6f}, Validation Loss: {val_loss.item():.6f}')

# Plot training and validation losses
plt.figure()
plt.plot(train_losses, label='Training Loss')
plt.plot(val_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Evaluate the model on the test set
model.eval()
with torch.no_grad():
    test_predictions = model(test_inputs)
    test_loss = criterion(test_predictions, test_targets)
    print(f'Test Loss: {test_loss.item():.6f}')
    test_predictions = test_predictions.numpy()
    test_targets = test_targets.numpy()

# Plot the true vs predicted values for the test set
plt.figure()
plt.scatter(test_targets, test_predictions, label='Predicted vs True')
plt.xlabel('True Values')
plt.ylabel('Predicted Values')
plt.legend()
plt.show()