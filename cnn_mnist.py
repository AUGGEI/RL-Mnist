import numpy as np
import torch
import torchvision
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST
from torchvision.utils import save_image
import matplotlib.pyplot as plt
from tensorboardX import SummaryWriter

#===========================================================
# Model Architecture 

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.feature1 = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=1, padding=1),  #  28 *28*16
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=2),  #  14*14*16

            nn.Conv2d(16, 32, 3, stride=1, padding=1),  #  14 *14 *32
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=2),  #   7 *7 *32

            nn.Conv2d(32, 64, 3, stride=1, padding=1),  # 7*7*64
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=2)
        )

        self.fc =nn.Sequential(
            nn.Linear(576,3136), 
            nn.ReLU(True)
        )
        self.classification = nn.Sequential(
            nn.Linear(3136, 10),
            nn.Softmax()
        )

    def forward(self, x):
        x = self.feature1(x)
        x = x.view(-1, 576)
        x = self.fc(x)
        x = self.classification(x)
        return x
    
    def _feature_(self, x):
        x = self.feature1(x)
        x = x.view(-1, 576)
        return x
    def fc_(self, x):
        x = self.fc(x)
        return x


# #===========================================================
# # To turn tensors into images

# def to_img(x):
#     x = (0.5 * x) + 0.5
#     return (x.view(x.shape[0],1,28,28))

# #===========================================================
# # Function to train model

# def train_model(model,dataloader_train,dataloader_test,num_epochs,cost,optimizer):
#     writer = SummaryWriter('log')
    

#     for epoch in range(1,num_epochs+1):
#         loss_values = []
#         for i, data in enumerate(dataloader_train) :
#             img, labels= data
#             img = Variable(img).cuda()
#             labels = Variable(labels).cuda()
#             # =================== forward =====================
#             output = model(img)
#             loss = cost(output, labels)
#             # =================== backward ====================
#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()
#         # =================== log ========================

#             loss_values.append(loss.data)
#             writer.add_scalar('Train/Loss', loss.data, i * 1/len(dataloader_train) + epoch - 1)

#         print('epoch [{}/{}], loss:{:.4f}'.format(epoch, 
#                                                  num_epochs, 
#                                                   loss.data))
            
        
#         correct = 0
#         _sum = 0

#         for idx, (test_x, test_label) in enumerate(dataloader_test):
#             test_x = test_x.cuda()
#             predict_y = model(test_x.float()).detach()
#             predict_ys = np.argmax(predict_y.cpu(), axis=-1)
#             label_np = test_label.numpy()
#             _ = predict_ys == test_label
#             correct += np.sum(_.numpy(), axis=-1)
#             _sum += _.shape[0]

#         print('accuracy: {:.4f}'.format(correct / _sum))
#         writer.add_scalar('Train/ACC', correct / _sum, epoch)

#         # if(epoch==1 or epoch%10==0):
#         #     pic = to_img(output.cpu().data)
#         #     save_image(pic, '/home/user/liuhongxing/convolutional_image_encoder/data/train_{}.png'.format(epoch))
        
#     return loss_values
   
# #===========================================================
# # Required tranforms

# img_transform = transforms.Compose([
#     transforms.ToTensor(),
#     transforms.Normalize(mean=(0.5,0.5,0.5), std= (0.5, 0.5,0.5))
# ])

# #===========================================================
# # Defined parameters

# EPOCHS = 100
# BATCH_SIZE = 128
# LEARNING_RATE = 0.001
# writer = SummaryWriter()
# #===========================================================
# # Dataset and DataLoader

# train_dataset = MNIST('./data',
#                       train = True,
#                       transform = img_transform,
#                       download = True)

# train_dataloader = DataLoader(train_dataset,
#                               batch_size = BATCH_SIZE,
#                               shuffle = True,
#                               num_workers = 4)

# test_dataset = MNIST('./data',
#                       train = False,
#                       transform = img_transform,
#                       download = True)

# test_dataloader = DataLoader(test_dataset,
#                               batch_size = BATCH_SIZE,
#                               shuffle = True)
# print('train_dataloader', len(train_dataloader))
# #===========================================================
# # Compiling the model

# model = CNN().cuda()
# cost = nn.CrossEntropyLoss()
# optimizer = torch.optim.Adam(model.parameters(),
#                              lr = LEARNING_RATE,
#                              weight_decay=1e-5)

#===========================================================
# Training the model

# loss = train_model(model,train_dataloader,test_dataloader,EPOCHS,cost,optimizer)

#===========================================================
# Saving the model

# torch.save(model.state_dict(),"cnn_mnist.pth")