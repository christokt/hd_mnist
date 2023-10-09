import torch
import torch.nn as nn
import torch.nn.functional as F

class HDCNN(nn.Module):
    def __init__(self, hd_dim):
        super(HDCNN, self).__init__()
        
        # Extract features from images using a CNN
        self.conv1 = nn.Conv2d(1, 24, kernel_size=3)

        # Project each feature map to hyperdimensional space
        self.hd_projection = nn.Linear(26 * 26, hd_dim) 

        # Freeze weights of linear layer
        for param in self.hd_projection.parameters():
            param.requires_grad = False
    
    def forward(self, x):
        x = F.relu(self.conv1(x))
        
        # Suppose x.shape is [batch_size, num_channels, height, width]
        # print(x.shape)
        batch_size, num_channels, height, width = x.shape
        
        # Flatten and project each feature map to hyperdimensional space
        x = x.view(batch_size, num_channels, -1)  # shape becomes [batch_size, num_channels, height * width]
        x_hd = [self.hd_projection(x[:, i, :]) for i in range(num_channels)]
        
        # Associate the hyperdimensional vectors
        x_hd = torch.stack(x_hd, dim=1)  # shape becomes [batch_size, num_channels, hd_dimension]
        x_hd = torch.sum(x_hd, dim=1)  # shape becomes [batch_size, hd_dimension]

        # Apply sign function to return a binary vector of 1s and -1s
        x_hd = torch.sign(x_hd)
        
        return x_hd
