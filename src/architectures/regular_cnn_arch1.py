from torch import nn

class Conv1dLinear(nn.Module):
  def __init__(self, input_channels, input_len, output_size):
    super().__init__()
    
    self.cnn_mpool_linear_relu_sigmoid = nn.Sequential(
      nn.Conv1d(input_channels, 64, kernel_size=5, padding=2),
      nn.MaxPool1d(kernel_size=2, stride=2),
      
      nn.Conv1d(64, 64, kernel_size=5, padding=2),
      nn.MaxPool1d(kernel_size=2, stride=2),

      nn.Flatten(),

      nn.Linear(64*(input_len // 2 // 2), 2048),
      nn.ReLU(),
      nn.Linear(2048, 2048),
      nn.ReLU(),
      nn.Linear(2048, output_size),
      nn.Sigmoid()
    )


  def forward(self, x):
    return self.cnn_mpool_linear_relu_sigmoid(x)
