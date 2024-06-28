import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

def hidden_init(layer):
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_in)
    return (-lim, lim)

class Actor(nn.Module):
    """Actor (Policy) Model."""
    def __init__(self, state_size, action_size, seed, fc1_units=400, fc2_units=300):
        """Initialize parameters and build model.
        Params
        ======
            state_size: Dimension of each non-image state
            action_size: Dimension of each action
            seed: Random seed
            fc1_units: Number of nodes in first hidden layer after concatenating CNN and state inputs
            fc2_units: Number of nodes in second hidden layer
        """
        super(Actor, self).__init__()
        self.seed = torch.manual_seed(seed)
        # CNN for image processing
        self.conv1 = nn.Conv2d(3, 16, kernel_size=5, stride=2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=2)
        self.conv4 = nn.Conv2d(64, 128, kernel_size=3, stride=2)

        self.fc1 = nn.Linear(state_size + 128 * 12 * 12, fc1_units)  # Adjust input size based on output of conv layers
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.fc3 = nn.Linear(fc2_units, action_size)
        self.reset_parameters()

    def reset_parameters(self):
        self.conv1.weight.data.uniform_(*hidden_init(self.conv1))
        self.conv2.weight.data.uniform_(*hidden_init(self.conv2))
        self.conv3.weight.data.uniform_(*hidden_init(self.conv3))
        self.conv4.weight.data.uniform_(*hidden_init(self.conv4))
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state, image):
        """Build an actor (policy) network that maps states -> actions."""
        image = F.relu(self.conv1(image))
        image = F.relu(self.conv2(image))
        image = F.relu(self.conv3(image))
        image = F.relu(self.conv4(image))
        image = image.view(-1, 128 * 12 * 12)  # Flatten the output for the FC layers

        state = torch.cat((image, state), dim=1)  # Concatenate flattened image and state

        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return torch.tanh(self.fc3(x))

class Critic(nn.Module):
    """Critic (Value) Model."""
    def __init__(self, state_size, action_size, seed, fcs1_units=400, fc2_units=300):
        """Initialize parameters and build model.
        Params
        ======
            state_size: Dimension of each non-image state
            action_size: Dimension of each action
            seed: Random seed
            fcs1_units: Number of nodes in the first hidden layer
            fc2_units: Number of nodes in the second hidden layer
        """
        super(Critic, self).__init__()
        self.seed = torch.manual_seed(seed)
        # CNN for image processing
        self.conv1 = nn.Conv2d(3, 16, kernel_size=5, stride=2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=2)
        self.conv4 = nn.Conv2d(64, 128, kernel_size=3, stride=2)

        self.fc1 = nn.Linear(state_size + 128 * 12 * 12, fc1_units)  # Adjust input size based on output of conv layers
        self.fc2 = nn.Linear(fcs1_units + action_size, fc2_units)
        self.fc3 = nn.Linear(fc2_units, 1)
        self.reset_parameters()

    def reset_parameters(self):
        self.conv1.weight.data.uniform_(*hidden_init(self.conv1))
        self.conv2.weight.data.uniform_(*hidden_init(self.conv2))
        self.conv3.weight.data.uniform_(*hidden_init(self.conv3))
        self.conv4.weight.data.uniform_(*hidden_init(self.conv4))
        self.fcs1.weight.data.uniform_(*hidden_init(self.fcs1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state, action, image):
        """Build a critic (value) network that maps (state, action) pairs -> Q-values."""
        image = F.relu(self.conv1(image))
        image = F.relu(self.conv2(image))
        image = F.relu(self.conv3(image))
        image = F.relu(self.conv4(image))
        image = image.view(-1, 128 * 12 * 12)  # Flatten the output for the FC layers

        state = torch.cat((image, state), dim=1)  # Concatenate flattened image and state

        xs = F.relu(self.fcs1(state))
        x = torch.cat((xs, action), dim=1)
        x = F.relu(self.fc2(x))
        return self.fc3(x)
