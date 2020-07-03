from torch.utils.data import Dataset
from PythonFiles.data_init import Data_Initialiser
import os
import pandas as pd
import pathlib
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms


class KeypointDataset(Dataset):

    def __init__(self,train=False,val=False,test=False,remake_file=True,transform=None):
        if remake_file:
            Data_Initialiser()
        if train:
            csv_file_name="train_data.csv"
            print("Train Data Loaded")
        elif val:
            csv_file_name="val_data.csv"
            print("Validation Data Loaded")
        elif test:    
            csv_file_name="test_data.csv" 
            print("Test Data Loaded")
        else:
            print("Please Check the Parameters")

        self.transform=transforms.Compose([transforms.ToTensor()])    

        root_path=pathlib.Path().absolute()
        data_dir=os.path.join(root_path,"Datasets" ,csv_file_name)
        self.data=pd.read_csv(data_dir)
        

    def _get_image(self,idx):
        img_string=self.data.loc[idx]['Image']
        img_np=np.array([int(item) for item in img_string.split()]).reshape(96,96)
        img_np=img_np#/255
        return img_np


    def _get_keypoints(self,idx):
        keypoint_columns=list(self.data.columns[1:-1])
        keypoint=self.data.iloc[idx][keypoint_columns]
        keypoint_np=np.array(keypoint).reshape(6,1).astype(int)
        keypoint_np=keypoint_np.reshape(3,2)
        keypoint_np=keypoint_np#/96
        return keypoint_np

    def show_plot(self,idx):
        img_np=self._get_image(idx)
        #img_np=img_np/255
        #img_np=img_np.reshape(96,96)
        keypoint_np=self._get_keypoints(idx)
        keypoint_np=keypoint_np.reshape(3,2)
        plt.imshow(img_np,cmap='gray')
        plt.scatter(keypoint_np[:,0],keypoint_np[:,1],marker='+')


    def __len__(self):
        return self.data.shape[0]


    def __getitem__(self,idx):
        img_np=self._get_image(idx)
        img_np=np.expand_dims(img_np, axis=2).astype(np.uint8)
        keypoint_np=self._get_keypoints(idx)
        keypoint_np=keypoint_np/96
        return {'image':self.transform(img_np),'keypoints':self.transform(keypoint_np)}


    def test_plot(self,idx,outputs_test):
        img_np=self._get_image(idx)
        keypoint_np=self._get_keypoints(idx)
        keypoint_np=keypoint_np.reshape(3,2)
        keypoint_np_pred=outputs_test*96
        plt.figure(figsize=(5,5))
        plt.imshow(img_np,cmap='gray')
        plt.scatter(keypoint_np[:,0],keypoint_np[:,1],marker='+',label='True Position')
        plt.scatter(keypoint_np_pred[:,0],keypoint_np_pred[:,1],marker='*',label='Predicted Position')
        plt.legend()
        plt.savefig(str(idx)+".png", bbox_inches='tight')






