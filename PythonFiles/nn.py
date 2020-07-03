"""Models for facial keypoint detection"""

import torch
import torch.nn as nn

class NNModel(nn.Module):

    def __init__(self,hparams):

        super(NNModel,self).__init__()

        self.hparams = hparams
        self.p=hparams["droupout_p"]

        self.conv1=nn.Conv2d(1,4,4,groups=1)
        self.activ1=nn.ELU()
        self.bn1=nn.BatchNorm2d(4)
        self.max1=nn.MaxPool2d(2, stride=1)
        
        self.conv2=nn.Conv2d(4,8,4,groups=1)
        self.activ2=nn.ELU()
        self.bn2=nn.BatchNorm2d(8)
        self.max2=nn.MaxPool2d(2, stride=1)
        
        self.conv3=nn.Conv2d(8,16,4)
        self.activ3=nn.ELU()
        self.bn3=nn.BatchNorm2d(16)
        self.max3=nn.MaxPool2d(2, stride=1)
        
        self.conv4=nn.Conv2d(16,32,4)
        self.activ4=nn.ELU()
        self.bn4=nn.BatchNorm2d(32)
        self.max4=nn.MaxPool2d(2, stride=1)
        

        self.conv5=nn.Conv2d(32,64,4)
        self.activ5=nn.ELU()
        self.bn5=nn.BatchNorm2d(64)
        self.max5=nn.MaxPool2d(2, stride=1)
        
        self.conv6=nn.Conv2d(64,128,4,stride=2)
        self.activ6=nn.ELU()
        self.bn6=nn.BatchNorm2d(128)
        self.max6=nn.MaxPool2d(2, stride=2)
        
        self.conv7=nn.Conv2d(128,256,4,stride=2)
        self.activ7=nn.ELU()
        self.bn7=nn.BatchNorm2d(256)
        self.max7=nn.MaxPool2d(2, stride=2)

        self.lin1=nn.Linear(4096,2048)
        self.activ_lin1=nn.ELU()
        self.bn_lin1=nn.BatchNorm1d(2048)
        self.dp_lin1=nn.Dropout(p=self.p)


        self.lin2=nn.Linear(2048,512)
        self.activ_lin2=nn.ELU()
        self.bn_lin2=nn.BatchNorm1d(512)
        self.dp_lin2=nn.Dropout(p=self.p)

        self.lin3=nn.Linear(512,6)


    def forward(self, x):

        pass

        x=self.conv1(x)
        x=self.activ1(x)
        x=self.bn1(x)
        x=self.max1(x)
        
        x=self.conv2(x)
        x=self.activ2(x)
        x=self.bn2(x)
        x=self.max2(x)
        
        x=self.conv3(x)
        x=self.activ3(x)
        x=self.bn3(x)
        x=self.max3(x)
        
        x=self.conv4(x)
        x=self.activ4(x)
        x=self.bn4(x)
        x=self.max4(x)
        

        x=self.conv5(x)
        x=self.activ5(x)
        x=self.bn5(x)
        x=self.max5(x)
        
        x=self.conv6(x)
        x=self.activ6(x)
        x=self.bn6(x)
        x=self.max6(x)
        
        x=self.conv7(x)
        x=self.activ7(x)
        x=self.bn7(x)
        x=self.max7(x)

        x=torch.flatten(x,1,3)

        x=self.lin1(x)
        x=self.activ_lin1(x)
        x=self.bn_lin1(x)
        x=self.dp_lin1(x)

        x=self.lin2(x)
        x=self.activ_lin2(x)
        x=self.bn_lin2(x)
        x=self.dp_lin2(x)

        x=self.lin3(x)

        return x


