# -*- coding: utf-8 -*-
"""TYG PYTORCH .ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1ZZ39H2UDxif6sOAJA_CFUS7PWIqu978s
"""

import torch

tensor1 = torch.tensor([1, 2, 3, 4])

print(tensor1)

tensor2 = torch.rand(2,3)

print(tensor2)

tensor3 = torch.zeros(3,3)

print(tensor3)

"""Broadcasting"""

result1 = tensor1 + 5

print(result1)

result2 = tensor2 *3

print(result2)

"""MAtrix Multiplication"""

matrix1 = torch.rand((2,3))
matrix2 = torch.rand((3,1))

# output should be 2x1
# 2 rows and 1 column
result3 = matrix1.matmul(matrix2)
print(result3)

from torch.utils.data import DataLoader, Dataset

class MyDataset(Dataset):
  def __init__(self):
    # 100 data points, 5 features
    # n_rows x n_cols
    self.data = torch.rand((100, 5))

    # labels would be integers between 0 and 1
    # So upper bound (non-inclusive) would be 2
    # low_range, hi_range, shape => 1D vector with 100 labels
    self.labels = torch.randint(0, 2,(100,))

  def __len__(self):
    return len(self.data)


  def __getitem__(self, idx):
    """

    Args:
      idx:

    Returns:

    """
    # return  data, corresponding label
    return self.data[idx], self.labels[idx]

dataset = MyDataset()
dataloader = DataLoader(dataset, batch_size = 10, shuffle = True)

# Iterate through the dataloader

for batch_idx, (features, labels) in enumerate(dataloader):
  print(f"Batch {batch_idx+1}")
  print("Features:", features)
  print("Labels:", labels)

"""Dealing with numbers and operations"""

# scalar value with gradient tracking
# torch keeps track of the variable in the computation graph

x = torch.tensor(2.0, requires_grad = True)
# x = torch.tensor(2.0, requires_grad = False)

z = torch.tensor([2.0, 4.1], requires_grad= True)


# compute a function [y = x^3]
# y = x ** 3
y = z[0]**2 + z[1]

# compute the gradient dy/dx
y.backward()

# use .item() for using the value instead of the object
# print("the gradient is", x.grad.item())

# convert .tolist() / .numpy()
print("the gradient is", z.grad)
print("the gradient is", z.grad.tolist())

"""Operations"""

# Tensor with values from 0 to 9
x = torch.arange(0,10).float()
y = x**2 + 3*x + 2

# efficiently eliminate loops
print("The value are:", y.tolist())

"""Broadcasting"""

# shape = matrix
x = torch.rand((3,4))

# shape = vector [1x4]
y = torch.rand((4,))

result = x + y
print(result.shape)


print(result.tolist())

import torch

"""Neural Network"""

import torch
import torch.nn as nn

class SimpleNet(nn.Module):
  def __init__(self, input_size, hidden_size, num_classes):
    # input shape => (batch_size = 10, num_features = 5)
    # Hence input_size = num_features

    # hidden_size = 50
    # more capacity to learn richer representations

    # output shape => (batch_size = 10, num_clases = 2)

    super(SimpleNet, self).__init__()
    self.fc1 = nn.Linear(input_size, hidden_size)
    self.fc2 = nn.Linear(hidden_size, num_classes)

  def forward(self, x):
    # relu adds non-linearity to
    x = torch.relu(self.fc1(x))
    x = self.fc2(x)
    return x

model = SimpleNet(5, 50, 2)

# batch of 10 samples, 5 features
example_input = torch.rand(10, 5)

# forward pass
output = model(example_input)



print("Input Shape: ", example_input.shape)
print("Output Shape: ", output.shape)

import torch

"""Install transformers"""

!pip install transformers

from transformers import BertTokenizer, BertForSequenceClassification

# Load tokenizer and model
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# Add a linear classification layer on top of BERT
model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels = 2)


# tokenize a sample sentence
text = "PyTorch is great for neural networks!"
inputs = tokenizer(text, return_tensors = "pt", padding = True, truncation = True, max_length = 512)

outputs = model(**inputs)
print("Logits: ", outputs.logits)

print(outputs.logits.tolist())

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

class MyDataset(Dataset):
  # def __init__(self, input_size / num_features, hidden_size, num_classes):
  def __init__(self):
    super(MyDataset, self).__init__()
    # 100 samples with 5 features each
    self.data = torch.rand(100, 5)

    # labeled 0,1
    self.labels = torch.randint(0, 2, (100,))


  def __len__(self):
    return len(self.data)


  def __getitem__(self, idx):
    # features, labels
    return self.data[idx], self.labels[idx]

"""CHECK FOR GPU PRESENCE"""

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tensor = torch.rand((2,2)).to(device)
print(tensor.shape, tensor.dtype, tensor.device)
tensor = torch.rand(2,2).to(device)
print(tensor.shape, tensor.dtype, tensor.device)

# train_dataloader = Dataloader(training_data, batch_size = 64, shuffle = True)

# train_features, train_labels = next(iter(train_dataloader))

"""OPENAI - QUESTIONS"""

import torch

## PROBLEM - 1

def scale_and_add(matrix, vector, scale):
  # matrix => size m x n

  # scale all the elements of matrix by scale

  matrix = matrix * scale

  # add vector to each row of the scaled matrix

  # broadcast
  matrix = matrix + vector

  return matrix


# matrix = torch.rand((1,2))

scale = torch.tensor(2)
vector = torch.tensor([1, 1])
matrix = torch.tensor([[1,2], [2,4]])
print(scale_and_add(matrix, vector, scale))

## PROBLEM  - 2

import torch
from torch.utils.data import DataLoader, Dataset


class SquareDataset(Dataset):
  def __init__(self, n):
    # num_samples = n, num_features = 1
    self.data = torch.randint(1, 100, (n,))
    self.labels = torch.tensor([x**2 for x in self.data])

  def __len__(self):
    return len(self.data)

  def __getitem__(self, idx):
    return self.data[idx], self.labels[idx]

n = 3
train_sd = SquareDataset(n)
train_loader = DataLoader(train_sd, batch_size = 5, shuffle = True)

# trainloader IS an iterator
train_loader = next(iter(DataLoader(train_sd, batch_size = 5, shuffle = True)))


print(train_loader[0])

features, labels = train_loader
print(features)
print(labels)

print(features.shape, features.dtype)

# PROBLEM - 3

import torch
import torch.nn as nn

class NeuralNet(nn.Module):
  def __init__(self, input_size, hidden_size, num_classes):
    super().__init__()
    self.fc1 = nn.Linear(input_size, hidden_size)
    self.fc2 = nn.Linear(hidden_size, num_classes)

  def forward(self, x):
    # nn.ReLU
    x = torch.relu(self.fc1(x))
    out = self.fc2(x)
    return out


model = NeuralNet(8, 32, 4)
data_point = torch.rand((16, 8))

out = model(data_point)

print(out.shape)

x = torch.tensor(2.0, requires_grad= True)

y = 3*x**3 + 2*x**2 + x

# compute dy/dx
y.backward()

print(x.grad)

x = torch.rand((3,))
x.requires_grad = True

y = x**2 + 3*x + 5

y.backward(torch.tensor([1.0, 1.0, 1.0]))

print(x.grad)

import torch

# torch.rand(size)
x = torch.rand(1)
print(x.shape, x.dtype)

"""CNNs"""

import torch
import torch.nn as nn

class ConvSigmoidNet(nn.Module):
  def __init__(self):
    super().__init__()

    # grayscale image => point[x,y] in the image has only single value
    self.conv1 = nn.Conv2d(in_channels = 1, out_channels = 1, kernel_size =3, stride = 1, padding  = 1)

    # Sigmoid is used for binary classification
    self.sigmoid = nn.Sigmoid()


  def forward(self,x):
    x = self.conv1(x)
    out = self.sigmoid(x)
    return out

# instantiate the model
model = ConvSigmoidNet()

# batch_size, channel, height, width
input_tensor = torch.rand(1, 1, 5, 5)


output = model(input_tensor)
print(input_tensor.shape)
print(output.shape)
print(output)
print(output.grad_fn)

"""RNN FOR SEQUENCE CLASSIFICATION"""



"""TYG-1]"""

import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# Define the neural network
class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)  # First layer
        self.fc2 = nn.Linear(hidden_size, num_classes)  # Output layer
        self.sigmoid = nn.Sigmoid()  # Activation for binary classification

    def forward(self, x):
        x = torch.relu(self.fc1(x))  # Apply ReLU to hidden layer
        out = self.sigmoid(self.fc2(x))  # Apply Sigmoid to output
        return out

# Generate synthetic data
torch.manual_seed(42)  # For reproducibility
data = torch.rand(100, 2)  # 100 samples, 2 features each
labels = torch.randint(0, 2, (100,)).float()  # Binary labels (0 or 1)

# Define model, loss, and optimizer
model = NeuralNet(input_size=2, hidden_size=10, num_classes=1)  # 1 output for binary classification
criterion = nn.BCELoss()  # Binary Cross-Entropy Loss
optimizer = optim.SGD(model.parameters(), lr=0.1)  # Stochastic Gradient Descent

# Training loop
losses = []
epochs = 100
for epoch in range(epochs):
    # Forward pass
    predictions = model(data)  # Predictions
    loss = criterion(predictions.squeeze(), labels)  # Calculate loss

    # Backward pass and optimization
    optimizer.zero_grad()  # Clear previous gradients
    loss.backward()  # Compute gradients
    optimizer.step()  # Update weights

    # Record loss
    losses.append(loss.item())
    if (epoch + 1) % 10 == 0:
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")

# Plot loss over epochs
plt.plot(losses)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss")
plt.show()

import torch
import torch.nn as nn
import torch.optim as optim

class NeuralNet(nn.Module):
  def __init__(self, input_size, hidden_size, num_classes):
    # init
    super().__init__()

    self.fc1 = nn.Linear(input_size, hidden_size)
    self.fc2 = nn.Linear(hidden_size, num_classes)

    # ReLU is used for hidden layers
    self.relu = nn.ReLU

    # Sigmoid is used for binary classification
    self.sigmoid = nn.Sigmoid()

  def forward(self, x):
    x = self.fc1(x)
    x = self.relu(x)
    x = self.fc2(x)
    out = self.sigmoid(x)

    return out

!pip install torch torchvision

import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader, Dataset

class MyDataset(Dataset):
  def __init__(self):
    self.data = []
    self.labels = labels


  def __len__(self):
    return len(self.data)

  def __getitem__(self, idx):
    return self.data[idx], self.labels[idx]

device = "cuda" if torch.cuda.is_available() else "cpu"

print(device)

# We have different shapes in PyTorch
# This shape is a vector [we got scalars and matrix as well]

tensor1 = torch.rand((5,))

print(tensor1.shape)
print(tensor1.dtype)
print(tensor1.device)

print(tensor1.tolist())

# define x as a scalar

x = torch.rand(1)

# set gradient accumulation to true
# Now x's gradients would be tracked in the computation graph
x.requires_grad = True

y = x**2 + 3

# compute dy/dx
y.backward()

print(x.grad)
print(x.shape)
print(x.dtype)
print(x.tolist())
print(x.ndim)

class NeuralNet(nn.Module):
  # input_layers [features] , hidden_layers [complexity], num_classes

  def __init__(self, input_size, hidden_size, num_classes):
    # nn.Linear + nn.ReLU
    # nn.Sequential
    self.fc1 = nn.Linear(input_size, hidden_size)
    self.fc2 = nn.Linear(hidden_size, num_classes)

    # Use a relu layer for learning between hidden layers
    # Or it will just be a linear transformation
    # Degrees of freedom of learning / complexity of the equation
    self.relu = nn.ReLU()


  # forward pass
  # loss.backward() for the backward pass => uodating the weights
  def forward(self, x):
    x = self.fc1(x)
    x = self.relu(x)
    x = self.fc2(x)
    out = self.relu(x)

    return out