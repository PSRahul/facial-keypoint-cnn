#File Generates the Training , Validation and Testing data
#from the orginal csv files

import os
import pathlib
import pandas as pd
import numpy as np

class Data_Initialiser():

    def __init__(self):

        #Training and Validation Data
        #Training Dataset Generation

        train_file_name="train_orginal.csv"
        train_root_path=pathlib.Path().absolute()
        train_data_dir=os.path.join(train_root_path,"Datasets" ,train_file_name)
        train_data=pd.read_csv(train_data_dir)
        deleted_keys=[]

        #Removes when number of samples are less than 7000
        for key in train_data.keys():
            if (train_data[key].count())<7000:
                deleted_keys.append(key)
                del train_data[key]
                
        del train_data["nose_tip_x"]
        del train_data["nose_tip_y"]
        deleted_keys.append("nose_tip_x")
        deleted_keys.append("nose_tip_y")        
        train_data=train_data.dropna()
        train_data=train_data.reset_index(drop=True) 

        #Validation Dataset Generation

        val_index=np.random.choice(7000,1000,replace=False)
        val_data = train_data.iloc[val_index] 
        train_data=train_data.drop(val_index)


        test_file_name="test_orginal.csv"
        test_root_path=pathlib.Path().absolute()
        test_data_dir=os.path.join(test_root_path,"Datasets" ,test_file_name)
        test_data=pd.read_csv(test_data_dir)
        for key in deleted_keys:
            del test_data[key]
            
        test_data=test_data.dropna()
        #test_data=test_data.reset_index(drop=True) 


        #Saves the Training , Validation and Testing data as .csv files
        root_path=pathlib.Path().absolute()
        train_savedir=os.path.join(root_path,"Datasets" ,"train_data.csv")
        train_data.to_csv(train_savedir)
        val_savedir=os.path.join(root_path,"Datasets" ,"val_data.csv")
        val_data.to_csv(val_savedir)
        test_savedir=os.path.join(root_path,"Datasets" ,"test_data.csv")
        test_data.to_csv(test_savedir)

        print("Data Initialisation Complete")