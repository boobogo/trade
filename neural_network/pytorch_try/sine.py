import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

# Generate data
x = np.linspace(0, 2 * np.pi, 1000)  # 1000 points between 0 and 2*pi
y = np.sin(x)  # sine of x

# Convert to PyTorch tensors
inputs = torch.tensor(x, dtype=torch.float32).unsqueeze(1)  # Shape (1000, 1)
targets = torch.tensor(y, dtype=torch.float32).unsqueeze(1)  # Shape (1000, 1)

# Define the neural network model
class SineNN(nn.Module):
    def __init__(self, input_size, hidden1_size, hidden2_size, output_size):
        super(SineNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden1_size)  # First fully connected layer
        self.relu = nn.ReLU()  # Activation function
        self.fc2 = nn.Linear(hidden1_size, hidden2_size) # Second fully connected layer
        self.fc3 = nn.Linear(hidden2_size, output_size) # Third fully connected layer

    def forward(self, x):
        out = self.fc1(x)  # Linear transformation
        out = self.relu(out)  # Apply ReLU activation
        out = self.fc2(out)  # Linear transformation
        out = self.relu(out) # Apply ReLU activation
        out = self.fc3(out)
        return out

# Hyperparameters
input_size = 1  # One input feature (x value)
hidden1_size = 64  # Number of neurons in the hidden1 layer
hidden2_size = 64  # Number of neurons in the hidden2 layer
output_size = 1  # One output feature (y value)
learning_rate = 0.01
num_epochs = 1000

# Create the model, define the loss function and the optimizer
model = SineNN(input_size, hidden1_size, hidden2_size, output_size)
criterion = nn.MSELoss()  # Mean Squared Error Loss
optimizer = optim.Adam(model.parameters(), lr=learning_rate)  # Adam optimizer

# Training loop
for epoch in range(num_epochs):
    # Forward pass
    outputs = model(inputs)
    loss = criterion(outputs, targets)  # Compute the loss

    # Backward pass and optimization
    optimizer.zero_grad()  # Clear the gradients
    loss.backward()  # Compute gradients
    optimizer.step()  # Update weights

    if (epoch+1) % 100 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.6f}')

# Plot the results
predicted = model(inputs).detach().numpy()  # Get the predicted values
plt.plot(x, y, label='True Sine')
plt.plot(x, predicted, label='Predicted Sine')
plt.legend()
plt.show()