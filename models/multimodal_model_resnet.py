import torch
from torchvision.models import mobilenet_v3_small
import torch.nn as nn
import torch.functional as F
from torchvision import models
from torchvision.models import ResNet50_Weights

class PMModel(nn.Module):
    def __init__(self,S5no2_num_features=20,S5so2_num_features=20,tabular_input_count=3,tabular_features=3,out=1):
        super(PMModel, self).__init__()

    
        in_features= S5no2_num_features + tabular_features+ S5so2_num_features
       

        num_channels=4
        # Load the pre-trained ResNet-18 model
        
        self.mobile = models.mobilenet_v3_small(models.MobileNet_V3_Small_Weights.DEFAULT)
        self.mobile.features[0][0] = nn.Conv2d(num_channels, 16, 3, 1, 1)
        self.mobile.classifier[3] = nn.Linear(1024,S5no2_num_features)
        
        # Modify the input layer to accept multispectral data
        #resnet34
        self.resnet34 = models.resnet34(weights=models.ResNet34_Weights)
        self.resnet34.conv1 = nn.Conv2d(num_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        num_features1 = self.resnet34.fc.in_features
        self.resnet34.fc = nn.Linear(num_features1,S5no2_num_features)
        in_features= self.resnet34.fc.out_features*2+tabular_input_count
      


        #for param in self.resnet343.parameters():
            #param.required_grad=False
        
        # Modify the output layer for your specific number of classes
        
        #self.resnet18= models.resnet18(pretrained=True)
        #num_features = self.resnet18.fc.in_features
        
        

        

        
       

        self.backboneS5no2=self.resnet34  
                                      
        self.backboneS5so2=self.resnet34
        
        self.tabular=nn.Sequential(
                    nn.Linear(tabular_input_count, 16),
                    nn.ReLU(),
                    nn.Linear(16, 32),
                    nn.ReLU(),
                    nn.Linear(32, tabular_features))
        self.fusion=nn.Sequential(
                    nn.Linear(in_features,128),
                    nn.ReLU(),
                    nn.Linear(128,64),
                    nn.Dropout(0.25),
                    nn.ReLU(),
                    nn.Linear(64, 16),
                    nn.Dropout(0.25),
                    nn.ReLU(),
                    nn.Linear(16, out))
            
    def forward(self,x1,x2,x3):
        x1=self.backboneS5no2(x1).float()
       
        x2=self.backboneS5so2(x2).float()
        
        #x3=self.tabular(x3).float()
        
        x=torch.cat((x1,x2,x3),dim=1)
        
        return self.fusion(x)




#model=PMModel(S5no2_num_features=100, S5so2_num_features=100,tabular_input_count=2,tabular_features=3,out=1)  
#print(model)


        