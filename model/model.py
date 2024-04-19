from torch import nn

class BreastCancerNN(nn.Module):
    
    def __init__(self, in_features: int, out_features: int) -> None:
        
        super().__init__()
        
        self.layer_1 = nn.Linear(in_features=in_features, out_features=out_features)
        
    def forward(self, x):
        
        return self.layer_1(x)