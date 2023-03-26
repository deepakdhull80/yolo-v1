import torch
import torchvision

class YoloLoss(torch.nn.Module):
    def __init__(self, s, b, n_class, lambda_coord, lambda_noobj):
        """_summary_

        Args:
            s (_type_): _description_
            b (_type_): _description_
            n_class (_type_): _description_
            lambda_coord (_type_): _description_
            lambda_noobj (_type_): _description_
        """
        super().__init__()
        self.s = s
        self.b = b
        self.n_class =n_class
        self.lambda_coord = lambda_coord
        self.lambda_noobj = lambda_noobj
    
    def forward(self,y,y_h):
        """YoloLoss function

        Args:
            y (torch.tensor): shape->(batch,patch,patch,5*b+n_class)
            y_h (torch.tensor): shape->(batch,patch,patch,5*b+n_class)
        """
        
        ### no object loss
        no_obj_mask = y[:,:,:,0] == 0
        loss_noobj = self.lambda_noobj * torch.sum(torch.pow(y[no_obj_mask] - y_h[no_obj_mask], 2))

        ### object found loss
        mask = ~no_obj_mask
        loss_obj = torch.sum(torch.pow(y[mask] - y_h[mask], 2))
        loss_obj, loss_noobj, loss_coord, loss_prob = 0, 0, 0, 0
        for i in range(self.s):
            for j in range(self.s):
                for k in range(self.b):
                    mask = y[:,i,j,k*5] == 1
                    s_y, s_y_h = y[mask], y_h[mask]
                    loss_obj += (s_y[:,i,j,k*5]- s_y_h[:,i,j,k*5]).pow(2).sum()
                    
                    loss_noobj += (y[~mask][:,i,j,k*5]- y_h[~mask][:,i,j,k*5]).pow(2).sum()
                    
                    loss_coord += torch.sum(
                        (s_y[:,i,j,k*5+1] - s_y_h[:,i,j,k*5+1]).pow(2) +
                        (s_y[:,i,j,k*5+2] - s_y_h[:,i,j,k*5+2]).pow(2) +
                        (s_y[:,i,j,k*5+3].pow(0.5) - s_y_h[:,i,j,k*5+3].pow(0.5)).pow(2) +
                        (s_y[:,i,j,k*5+4].pow(0.5) - s_y_h[:,i,j,k*5+4].pow(0.5)).pow(2)
                    )

                    loss_prob += (s_y[:,i,j,k*5+5:] - s_y_h[:,i,j,k*5+5:].softmax(dim=0)).pow(2).sum()

        loss = loss_obj \
            + self.lambda_noobj * loss_noobj \
                + self.lambda_coord * loss_coord \
                    + loss_prob
        loss = loss / y.shape[0]

        return loss

