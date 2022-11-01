import torch
import torchvision

class YoloLoss(torch.nn.Module):
    def __init__(self, s, b, n_class, lambda_coord, lambda_noobj):
        super().__init__()
        self.s = s
        self.b = b
        self.n_class =n_class
        self.lambda_coord = lambda_coord
        self.lambda_noobj = lambda_noobj
    
    def forward(self,y,y_h):
        return

