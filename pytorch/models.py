"""Define the convolutional neural network architecture."""

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as I


class Net(nn.Module):
    def __init__(self):
        """Define all the layers of this CNN:
        - It takes in a batch of square grayscale images 224x224 as input
        - It consists of 3 combinations of Conv2d + MaxPool2d + Dropout layers
        - It ends with a linear layer that represents the keypoints. This last
        layer outputs 136 values, 2 for each of the 68 keypoint (x, y) pairs.
        """
        super(Net, self).__init__()
        # Convo2d layers with a 5x5 square convolution kernel
        self.conv1 = nn.Conv2d(1, 32, 5, padding=1) 
        self.conv2 = nn.Conv2d(32, 64, 5, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 5, padding=1)

        # Maxpooling layer that uses a square window of kernel_size=2, stride=2
        self.pool = nn.MaxPool2d(2, 2)

        # Dropout layers for regularization
        self.dropout1 = nn.Dropout(0.2)
        self.dropout2 = nn.Dropout(0.3)
        self.dropout3 = nn.Dropout(0.4)

        # Fully-connected layers
        self.fc1 = nn.Linear(128 * 26 * 26, 512)
        self.fc2 = nn.Linear(512, 136)       
        
    def forward(self, x):
        """Define the feedforward behavior of this model. x is a batch of input
        images and has dimension: batch_size * channels * height * width. We 
        can calculate output size (width and height) of an image at each step 
        using the following formula:
        output size = (image_size + padding * 2 - kernel_size)/stride + 1
        For example the output size of self.pool(F.relu(self.conv1(x))) is:
        conv1 output = (224 + 1 * 2 - 5)/1 + 1 = 222 --> 222 x 222 image
        pooling output = (222 + 0 * 2 - 2)/2 + 1 = 111 --> 111 x 111 image
        """
        x = self.pool(F.relu(self.conv1(x))) # torch.Size([1, 32, 111, 111])
        x = self.dropout1(x)
        x = self.pool(F.relu(self.conv2(x))) # torch.Size([1, 64, 54, 54])
        x = self.dropout1(x)
        x = self.pool(F.relu(self.conv3(x))) # torch.Size([1, 128, 26, 26])
        x = self.dropout2(x)
        x = x.view(-1, 128 * 26 * 26) # torch.Size([1, 86528])
        x = F.relu(self.fc1(x)) # torch.Size([1, 512])
        x = self.dropout3(x)
        x = self.fc2(x) # torch.Size([1, 136])
        return x

if __name__ == "__main__":
    # Create a 1-image batch to test the Net
    image = Variable(torch.randn(1, 1, 224, 224))
    net = Net()
    # To verify output size, add print(x.size()) at each step in forward
    net.forward(image)