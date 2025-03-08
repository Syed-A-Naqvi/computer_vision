import torch
import torch.nn as nn
import torch.nn.functional as F

class MyMLP(nn.Module):
    def __init__(self):
        super(MyMLP, self).__init__()
        
        # ----------------------------------------------------------
        # Input: 32×32×3 = 3072 elements per CIFAR-10 image.
        # Architecture:
        #   fc1: 3072 -> 2048
        #   fc2: 2048 -> 1024
        #   fc3: 1024 -> 512
        #   fc4: 512  -> 256
        #   fc5: 256  -> 10
        #
        # Applying a 20% dropout in four hidden layers when if enabled.
        # ----------------------------------------------------------

        self.fc1 = nn.Linear(3 * 32 * 32, 2048)
        self.drop1 = nn.Dropout(0.2)  # 20% dropout
        self.fc2 = nn.Linear(2048, 1024)
        self.drop2 = nn.Dropout(0.2)
        self.fc3 = nn.Linear(1024, 512)
        self.drop3 = nn.Dropout(0.2)
        self.fc4 = nn.Linear(512, 256)
        self.drop4 = nn.Dropout(0.2)
        self.fc5 = nn.Linear(256, 10)

    def forward(self, x, dropout=True):
        # Flatten image from (N, 3, 32, 32) -> (N, 3072)
        x = torch.flatten(x, start_dim=1)

        # fc1 -> ReLU
        x = F.relu(self.fc1(x))
        if dropout:
            x = self.drop1(x)

        # fc2 -> ReLU
        x = F.relu(self.fc2(x))
        if dropout:
            x = self.drop2(x)

        # fc3 -> ReLU
        x = F.relu(self.fc3(x))
        if dropout:
            x = self.drop3(x)

        # fc4 -> ReLU
        x = F.relu(self.fc4(x))
        if dropout:
            x = self.drop4(x)

        # Output layer (raw logits)
        x = self.fc5(x)
        return x