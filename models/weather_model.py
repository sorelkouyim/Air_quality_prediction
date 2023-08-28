import torch
from torchvision.models import mobilenet_v3_small
import torch.nn as nn
import torch.functional as F

class PMModelSingle(nn.Module):
    def __init__(self,tabular_input_count=2,tabular_features=1):
        super(PMModelSingle, self).__init__()

        
        
        
        self.tabular=nn.Sequential(
                    nn.Linear(tabular_input_count, 3),
                    nn.ReLU(),
                    nn.Linear(3, tabular_features))
                   
        
            
    def forward(self,x3):
       
        x3=self.tabular(x3).float()
        
        #x=torch.cat((x1,x2,x3),dim=1)
        
        return x3


#model=PMModelSingle(tabular_input_count=2,tabular_features=3)  

#print(model)


        