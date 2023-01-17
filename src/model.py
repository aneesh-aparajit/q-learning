import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms as T
from typing import Any
from PIL import Image
import numpy as np

class Transforms:
    def __init__(self) -> None:
        self.transform = T.Compose([
            T.Grayscale(num_output_channels=1),
            T.ToTensor(),
            T.Normalize([0.5,], [0.5,])
        ])
    
    def __call__(self, x: Image, *args: Any, **kwds: Any) -> Any:
        return self.transform(x)


class BreakoutModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 8, 5, 2, 0),
            nn.MaxPool2d(2), 
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(8, 1, 5, 2, 0),
            nn.MaxPool2d(2), 
            nn.ReLU()
        )
        self.linear = nn.Linear(108, 2)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.conv2(x)
        x = torch.flatten(x, start_dim=0)
        x = self.linear(x)
        x = F.tanh(x)
        return x


if __name__ == '__main__':
    x = Image.fromarray(np.uint8(np.random.randn(210, 160, 3)*255))
    transforms = Transforms()
    x = transforms(x)
    print(x.shape)
    model = BreakoutModel()
    y = model.forward(x)
    print(y)
