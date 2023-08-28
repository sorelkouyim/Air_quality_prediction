import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from PIL import Image



class AirPollutionDataset(Dataset):
    def __init__(self, images_dir,csv_file,transform=None):
        super()
        self.data_csv=pd.read_csv(csv_file,sep=',')
        #print(self.data_csv.head())
        self.images_dir=images_dir
        self.transform=transform
   
    def __len__(self):
        return len(self.data_csv)
    def __getitem__(self, idx) :
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name_S5no2 = os.path.join(self.images_dir,
                                self.data_csv.iloc[idx]['S5no2'])
        image_S5no2 = Image.open(img_name_S5no2).convert("RGBA")
        

        img_name_S5so2 = os.path.join(self.images_dir,
                                self.data_csv.iloc[idx]['S5so2'])
        image_S5so2 = Image.open(img_name_S5so2).convert("RGBA")
        
        if self.transform:
           
           image_S5no2= self.transform(image_S5no2)
           image_S5so2= self.transform(image_S5so2)
           #print(image_S5no2)
        

        label=torch.tensor(self.data_csv.iloc[idx]['P2'])
        
       
        


        tabular= self.data_csv.loc[idx,['humidity','temperature','co','no2','o3']]
        tabular = torch.tensor(np.array([tabular], dtype=float).reshape(-1, 5).flatten())
        sample = {'S5no2': image_S5no2, 'S5so2': image_S5so2,'tabular': tabular ,'label':label}
        return sample['S5no2'],sample['S5so2'],sample['tabular'],sample['label']
    def display_S5no2(self,idx,type='S5no2'):
        img_name_S5no2 = os.path.join(self.images_dir,
                                self.data_csv.iloc[idx][type])
        plt.imshow(Image.open(img_name_S5no2))

        plt.title("Sentinel5 data")
        plt.show()


        
def test():
    # create the multinomial dataset
    #transforms
    transform =transforms.Compose([
        
        transforms.Resize((400,400))]
        
    )
    dataset=AirPollutionDataset("data_dir",'multimodal_dataset.csv',transform=transform)
    # pint the first item
    print(dataset.__getitem__(1))
    # display the sentinel 5no2 for the second element
    dataset.display_S5no2(2,'S5no2')
    print(dataset.__getitem__(1).size)


    
#test the multimodel dataset creation       
#test()


        
        
