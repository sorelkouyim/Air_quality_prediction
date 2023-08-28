import torch
from torchvision.models import mobilenet_v3_small
import torch.nn as nn
import torch.functional as F

class PMModel(nn.Module):
    def __init__(self,S5no2_num_features=1000,S5so2_num_features=1000,tabular_input_count=4,tabular_features=1,out=1):
        super(PMModel, self).__init__()

        #self.S5no2_num_features=S5no2_num_features
        #self.tabular_input_count=tabular_input_count
        #self.tabular_features=tabular_features
        #self.S5so2_num_features= S5so2_num_features
        in_features= S5no2_num_features + tabular_features+ S5so2_num_features
        #self.out=out
        
        
        self.backboneS5no2=nn.Sequential(nn.Conv2d(4, 10, 3),
                              nn.BatchNorm2d(10), 
                              nn.ReLU(),
                              nn.MaxPool2d(3),
                              nn.Conv2d(10, 15, 3),
                              nn.BatchNorm2d(15), 
                              nn.ReLU(),
                              nn.MaxPool2d(3),
                              nn.Conv2d(15,20, 3),
                              nn.BatchNorm2d(20),
                              nn.ReLU(),
                              nn.MaxPool2d(3),
                              nn.Flatten(),
                              nn.Linear(980 , S5no2_num_features))         
                                      
        self.backboneS5so2=nn.Sequential(nn.Conv2d(4, 10, 3),
                              nn.BatchNorm2d(10),
                              nn.ReLU(),
                              nn.MaxPool2d(3),
                              nn.Conv2d(10,15, 3),
                              nn.BatchNorm2d(15),
                              nn.ReLU(),
                              nn.MaxPool2d(3),
                              nn.Conv2d(15,20, 3),
                              nn.BatchNorm2d(20),
                              nn.ReLU(),
                              nn.MaxPool2d(3),
                              nn.Flatten(),
                              nn.Linear(980 , S5so2_num_features))
        
        self.tabular=nn.Sequential(
                    nn.Linear(tabular_input_count, 16),
                    nn.ReLU(),
                    nn.Linear(16, 32),
                    nn.ReLU(),
                    nn.Linear(32, tabular_features))
        self.fusion=nn.Sequential(
                    nn.Linear(in_features, 128),
                    nn.ReLU(),
                    nn.Linear(128, 64),
                    nn.Dropout(0.25),
                    nn.ReLU(),
                    nn.Linear(64, 16),
                    nn.Dropout(0.25),
                    nn.ReLU(),
                    nn.Linear(16, out))
            
    def forward(self,x1,x2,x3):
        x1=self.backboneS5no2(x1).float()
        
        x2=self.backboneS5so2(x2).float()
      
        x3=self.tabular(x3).float()
        
        x=torch.cat((x1,x2,x3),dim=1)
        
        return self.fusion(x)




#model=PMModel(S5no2_num_features=1000, S5so2_num_features=1000,tabular_input_count=2,tabular_features=3,out=1)  

#print(model)


        