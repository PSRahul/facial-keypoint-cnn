#A wrapper Function for training the model
def trainer_function(hparams):
    from torch.utils.tensorboard import SummaryWriter
    from datetime import datetime
    from torch.utils.data import DataLoader
    import torch
    from torchvision import transforms
    import pathlib
    from PythonFiles.data_manip import KeypointDataset
    from PythonFiles.nn import NNModel
    import os

 
    #Loading Training and Validation Data
    transform=transforms.Compose([transforms.ToTensor()])
    train_data=KeypointDataset(train=True,val=False,test=False,remake_file=True,transform=transform)
    val_data=KeypointDataset(train=False,val=True,test=False,remake_file=False,transform=transform)
    print("Number of training samples:", len(train_data))
    print("Number of validation samples:", len(val_data))
    train_dataloader = DataLoader(train_data, batch_size=hparams["batch_size"])
    val_dataloader = DataLoader(val_data, batch_size=hparams["batch_size"])
    str_temp="adadelta"

    #Initialising Model
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Using the  Device - " ,device)
    model = NNModel(hparams)
    model=model.to(device)

    #Initialising Losses
    running_loss=0
    running_loss_val=0

    
    #Weight Initialisation
    def init_weights(m):
        if type(m) == torch.nn.Linear:
            torch.nn.init.kaiming_normal_(m.weight,nonlinearity='leaky_relu')
            m.bias.data.fill_(0.01)
            
        if type(m)== torch.nn.Conv2d:
            torch.nn.init.xavier_normal_(m.weight)
          
    model=model.apply(init_weights)

    #Optimiser and Loss function definition
    optimizer=torch.optim.Adadelta(model.parameters())
    loss_function=torch.nn.MSELoss()

    
    #Setting up Tensorboard
    logdir = "logs/" + datetime.now().strftime("%Y%m%d-%H%M%S")
    logdir = "logs/" + "p"+str(hparams["droupout_p"])+"epoch"+str(hparams["epochs"])+str_temp
    print("Saved Log Name ",logdir)
    writer = SummaryWriter(logdir)
    print("Started Training")
    
    #Training
    epochs=hparams["epochs"]
    for epoch in range(epochs):
        
        model.train()
        for i,data in enumerate(train_dataloader,0):
            
            #Loading the training data and sending to required device
            inputs=data["image"]
            labels=data["keypoints"]
            labels=labels.view(-1,6)
            labels=labels.type(dtype=torch.float)
            inputs=inputs.type(dtype=torch.float)
            inputs=inputs.to(device)
            labels=labels.to(device)
            
            optimizer.zero_grad()
            outputs=model(inputs.float()).float()
            loss=loss_function(outputs,labels)
            running_loss=loss.item()
            loss.backward()
            optimizer.step()
            
        
      
        running_loss=running_loss
        
        
        writer.add_scalar('Training loss',running_loss,epoch)
        
        #Validation Loss Calculation
        model.eval()
        with torch.no_grad():
            for i,data_val in enumerate(val_dataloader,0):
                inputs_val=data_val["image"]
                inputs_val=inputs_val.to(device)
                labels_val=data_val["keypoints"]
                labels_val=labels_val.view(-1,6) 
                labels_val=labels_val.to(device)
                outputs_val=model(inputs_val)
                loss_val=loss_function(outputs_val,labels_val)
                running_loss_val=loss_val.item()
                

        
        running_loss_val=running_loss_val
        writer.add_scalar('Validation loss',
                                    running_loss_val,
                                epoch )

            
        
        print("[Epoch %d]" % (epoch+1) , "Training Loss" ,'{0:1.4e}'.format(running_loss), " Validation Loss " ,'{0:1.4e}'.format(running_loss_val))
        
    print("Training Complete")  

    #Saving the Model  
    model_root_path=pathlib.Path().absolute()
    model_file_name="p="+str(hparams["droupout_p"])+"epoch="+str(hparams["epochs"])+str_temp
    model_data_dir=os.path.join(model_root_path,"Models" ,model_file_name)
    print("Saved Model Name " ,model_file_name)
    torch.save(model.state_dict(), model_data_dir)
    print("Model Saved")