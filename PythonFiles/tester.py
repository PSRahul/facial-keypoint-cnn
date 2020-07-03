#A wrapper function for implementing testing of samples
def tester_function(idx,model_file_name):

    from torch.utils.tensorboard import SummaryWriter
    from datetime import datetime
    from torch.utils.data import DataLoader
    import torch
    from torchvision import transforms
    import pathlib
    from PythonFiles.data_manip import KeypointDataset
    from PythonFiles.nn import NNModel
    import os

    
    #Model Directory
    model_root_path=pathlib.Path().absolute()
    model_data_dir=os.path.join(model_root_path,"Models" ,model_file_name)


    #Load Model
    hparams = {"batch_size" :32,
            "epochs":60,
            "droupout_p":0}
    model = NNModel(hparams)
    model.load_state_dict(torch.load(model_data_dir))
    model.eval()          


    #Load Test Data and set the device
    transform=transforms.Compose([transforms.ToTensor()])
    test_data=KeypointDataset(train=False,val=False,test=True,remake_file=False,transform=transform)
    test_dataloader = DataLoader(test_data, 100)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Using the Following Device - " ,device)
    model=model.to(device)


    #Calculating the test loss and test outputs
    loss_function=torch.nn.MSELoss()
    with torch.no_grad():
        for i,data_test in enumerate(test_dataloader,0):
            inputs_test=data_test["image"]
            inputs_test=inputs_test.to(device)
            labels_test=data_test["keypoints"]
            labels_test=labels_test.view(-1,6) 
            labels_test=labels_test.to(device)
            outputs_test=model(inputs_test)
            loss_test=loss_function(outputs_test,labels_test)
            running_loss_test=loss_test.item()#/(len(stest_data)*10000)

    #Plot the requested index 
    keypoint_pred=outputs_test[idx].reshape(3,2)
    keypoint_pred=keypoint_pred.to('cpu')
    keypoint_pred=keypoint_pred.detach().numpy()
    test_data.test_plot(idx,keypoint_pred)
    