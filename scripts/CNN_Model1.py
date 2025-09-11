"""
Script containing the model M1-VGG implementation
"""

import torch
from torch import nn

class CNN_Model1(nn.Module):
  """
  Model ...
  """
  def __init__(self, input_shape: int,hidden_units: int, output_shape:int):
    super().__init__()
    self.conv_block_1 = nn.Sequential(
        nn.Conv2d(in_channels=input_shape,
                  out_channels=hidden_units,
                  kernel_size=3,
                  stride=1,
                  padding=0),
        nn.ReLU(),
        nn.Conv2d(
            in_channels=hidden_units,
            out_channels=hidden_units,
            kernel_size=3,
            stride=1,
            padding=0
        ),
        nn.ReLU(),
        nn.MaxPool2d(
            kernel_size=2,
            stride=2
        )
    )
    self.conv_block_2 = nn.Sequential(
        nn.Conv2d(in_channels=hidden_units,
                  out_channels=hidden_units,
                  kernel_size=3,
                  stride=1,
                  padding=0),
        nn.ReLU(),
        nn.Conv2d(
            in_channels=hidden_units,
            out_channels=hidden_units,
            kernel_size=3,
            stride=1,
            padding=0
        ),
        nn.ReLU(),
        nn.MaxPool2d(
            kernel_size=2,
            stride=2
        )
    )
    self.pooling = nn.Sequential(
        nn.Flatten(),
        nn.Linear(
            in_features=hidden_units*13*13,
            out_features=output_shape
        )
    #self.pooling = nn.Sequential(nn.Flatten(), nn.LazyLinear(out_features=output_shape)) out maticlay get the in_fitures
    )

  def forward(self,x):
    #print(x.shape)
    #x=self.conv_block_1(x)
    #print(x.shape)
    #x=self.conv_block_2(x)
    #print(x.shape)
    #x=self.pooling(x)
    #print(x.shape)
    #return x
    return self.pooling(self.conv_block_2(self.conv_block_1(x)))


class CNN_Model1_224(nn.Module):
  """
  Model ...
  """
  def __init__(self, input_shape: int,hidden_units: int, output_shape:int):
    super().__init__()
    self.conv_block_1 = nn.Sequential(
        nn.Conv2d(in_channels=input_shape,
                  out_channels=hidden_units,
                  kernel_size=3,
                  stride=1,
                  padding=0),
        nn.ReLU(),
        nn.Conv2d(
            in_channels=hidden_units,
            out_channels=hidden_units,
            kernel_size=3,
            stride=1,
            padding=0
        ),
        nn.ReLU(),
        nn.MaxPool2d(
            kernel_size=2,
            stride=2
        )
    )
    self.conv_block_2 = nn.Sequential(
        nn.Conv2d(in_channels=hidden_units,
                  out_channels=hidden_units,
                  kernel_size=3,
                  stride=1,
                  padding=0),
        nn.ReLU(),
        nn.Conv2d(
            in_channels=hidden_units,
            out_channels=hidden_units,
            kernel_size=3,
            stride=1,
            padding=0
        ),
        nn.ReLU(),
        nn.MaxPool2d(
            kernel_size=2,
            stride=2
        )
    )
    self.pooling = nn.Sequential(
        nn.Flatten(),
        nn.Linear(
            in_features=hidden_units*53*53,
            out_features=output_shape
        )
    #self.pooling = nn.Sequential(nn.Flatten(), nn.LazyLinear(out_features=output_shape)) out maticlay get the in_fitures
    )

  def forward(self,x):
    # print(x.shape)
    # x=self.conv_block_1(x)
    # print(x.shape)
    # x=self.conv_block_2(x)
    # print(x.shape)
    # x=self.pooling(x)
    # print(x.shape)
    # return x
    return self.pooling(self.conv_block_2(self.conv_block_1(x)))

