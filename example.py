import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

# Define a simple neural network
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(2, 3)  # 2 input features, 3 output classes
    
    def forward(self, x):
        return self.fc1(x)

# Create synthetic data
X = torch.tensor([[0.1, 0.2], [0.2, 0.3], [0.3, 0.4], [0.4, 0.5]], dtype=torch.float32)
y = torch.tensor([0, 1, 2, 1], dtype=torch.long)

# Instantiate the model, loss function, and optimizer
model = SimpleNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.1)

# Train the model
num_epochs = 100
for epoch in range(num_epochs):
    model.train()
    
    optimizer.zero_grad()
    outputs = model(X)
    loss = criterion(outputs, y)
    loss.backward()
    optimizer.step()

    if (epoch+1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# Save the final parameters
final_params = [param.clone() for param in model.parameters()]

# Define a function to compute the loss for a given set of parameters
def compute_loss(params):
    with torch.no_grad():
        for p, fp in zip(model.parameters(), params):
            p.copy_(fp)
        outputs = model(X)
        loss = criterion(outputs, y)
    return loss.item()

# Generate random directions in parameter space
direction_1 = [torch.randn_like(p) for p in model.parameters()]
direction_2 = [torch.randn_like(p) for p in model.parameters()]

# Normalize directions
norm_1 = np.sqrt(sum(torch.sum(d ** 2).item() for d in direction_1))
norm_2 = np.sqrt(sum(torch.sum(d ** 2).item() for d in direction_2))
direction_1 = [d / norm_1 for d in direction_1]
direction_2 = [d / norm_2 for d in direction_2]

# Create a grid of alpha and beta values
alphas = np.linspace(-1, 1, 50)
betas = np.linspace(-1, 1, 50)
losses = np.zeros((len(alphas), len(betas)))

for i, alpha in enumerate(alphas):
    for j, beta in enumerate(betas):
        interpolated_params = [fp + alpha * d1 + beta * d2 for fp, d1, d2 in zip(final_params, direction_1, direction_2)]
        loss = compute_loss(interpolated_params)
        losses[i, j] = loss

# Plot the loss landscape in 3D
alpha_grid, beta_grid = np.meshgrid(alphas, betas)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(alpha_grid, beta_grid, losses, cmap='viridis')
ax.set_xlabel('Alpha')
ax.set_ylabel('Beta')
ax.set_zlabel('Loss')
ax.set_title('3D Loss Landscape')
plt.show()
