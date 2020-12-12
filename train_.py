# -*- coding: utf-8 -*-
import numpy as np
import torch
import torchvision
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from torchvision.datasets import MNIST
from torchvision.utils import save_image
import matplotlib.pyplot as plt
from tensorboardX import SummaryWriter
from deep_q_network import DeepQNetwork
from cnn_mnist import CNN
from sklearn.neighbors import NearestNeighbors,KNeighborsClassifier
import pickle
import cv2
from PIL import ImageFont, ImageDraw, Image
from tensorboardX import SummaryWriter

def rotation(degree):
    return transforms.RandomAffine(degrees=degree)

def step(imgs, actions):
    rotated_imgs = None
    for img, act in zip(imgs, actions):
        if act == 0:
            img = rotation((10,10.2))(transforms.ToPILImage()(img))
        elif act == 1:
            img = rotation((15,16))(transforms.ToPILImage()(img))
        elif act == 2:
            img = rotation((360-10,351))(transforms.ToPILImage()(img))
        elif act == 3:
            img = rotation((360-15,344))(transforms.ToPILImage()(img))
        elif act == 4:
            img = rotation(0)(transforms.ToPILImage()(img))
        else:
            print('ValueError')
        img = transforms.ToTensor()(img)
        if rotated_imgs is None:
            rotated_imgs = img.unsqueeze(0)
        else:
            rotated_imgs = torch.cat((rotated_imgs,img.unsqueeze(0)))
    # img.show()
    return rotated_imgs

def genenrate_D(imgs,labels):
    img = Variable(imgs).cuda()
    feature = cnn._feature_(img)
    feature = cnn.fc_(feature)
    feature = feature.cpu().detach().numpy()
    # print(feature[:-10, :].shape)
    KNN = KNeighborsClassifier(n_neighbors=5)
        # KNN = NearestNeighbors()
    # KNN.fit(feature[:-10, :], labels[:-10])
    # KNN.fit(feature, labels)
    
    #读取Model
    with open('KNeighborsClassifier.pickle', 'rb') as f:
        KNN = pickle.load(f)
    probility = KNN.predict_proba(feature)
    score = KNN.score(feature, labels)
    label_pre = KNN.predict(feature)
    # print('predict is {}, truth is {}, Accuracy is {}, probility is {}'.format(label_pre, labels[-10:], score, np.argmax(probility,axis=1)))
    # print('probility is {}'.format(np.argmax(probility,axis=1)))
    
    # with open('KNeighborsClassifier.pickle', 'wb') as f:
    #     pickle.dump(KNN, f)
     
    D = 1 - np.max(probility,axis=1)
    return D, label_pre, score
 

SEED = 1234
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)


#===========================================================
# Dataset and DataLoader
BATCH_SIZE =128

img_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5,0.5,0.5), std= (0.5, 0.5,0.5))
])

train_dataset = MNIST('./data',
                      train = True,
                      transform = img_transform,
                      download = True)

# train_db, val_db = random_split(train_dataset, [6000, 54000])
# val_db, _ = random_split(val_db,[1000,53000])

# train_db, val_db = random_split(train_dataset, [2000, 58000])
# val_db, _ = random_split(val_db,[1000,57000])

train_db, val_db = random_split(train_dataset, [600, 59400])
val_db, _ = random_split(val_db,[1000,58400])

train_dataloader = DataLoader(train_db, #train_dataset
                              batch_size = BATCH_SIZE,
                              shuffle = True,
                              num_workers = 4)

validation_dataloader = DataLoader(val_db,
                              batch_size = BATCH_SIZE,
                              shuffle = True,
                              num_workers = 4)

test_dataset = MNIST('./data',
                      train = False,
                      transform = img_transform,
                      download = True)
# print(len(test_dataset))
test_1, test_2 = random_split(test_dataset, [2000, 8000])
test_dataloader = DataLoader(test_1, # 
                              batch_size = BATCH_SIZE,
                              shuffle = True)
#===========================================================

cnn = CNN().cuda()
cnn.load_state_dict(torch.load("/home/user/liuhongxing/Mnist_RL/cnn_mnist.pth"))
Q = DeepQNetwork()
Q.load_state_dict(torch.load('/home/user/liuhongxing/Mnist_RL/Q_network_exploitation_600_act5.pth'))
optimizer = torch.optim.Adam(Q.parameters(), lr=1e-3)
criterion = nn.MSELoss()
threshold = 0.1
iterations = 70
gamma = 0.9

# writer = SummaryWriter('./log/Q_network_600_act5')
# # training and validation
# print('start Q_network_600_act5 training----------------------')
# for st, data in enumerate(train_dataloader):
#         imgs, labels = data
#         D, pre_label, _ = genenrate_D(imgs,labels)            # 生成图片与聚类中心距离D
#         reward = np.random.randint(1, size=(imgs.shape[0],1)) # 奖励初始化
#         Accuracy = []
#         D1, D2Label,_ = genenrate_D(imgs,labels)                     # 生成初始图片与聚类中心距离D1
#         state = torch.tensor(D1 - D).unsqueeze(1).float() # + 0.001    # 得到当前状态 S
#         for i in range(iterations):
#             # print(state[:10,:])
#             q_values = Q(state)                               # 得到当前Q值
#             # reward
#             reward[np.where(state.numpy() > threshold)] = -1
#             reward[np.where(state.numpy() < (0-threshold))] = 1 # 得到当前奖励
            
#             # Exploration or exploitation                        # 得到动作
#             if np.random.uniform() > 0.9:
#                 action = np.random.randint(5,size=(imgs.shape[0])) # 4
#             else:
#                 action = torch.argmax(q_values,dim=1)

#             rotation_imgs= step(imgs, action)                           # 执行动作
#             D_next, _ ,_= genenrate_D(rotation_imgs, labels)            # 得到下一时刻图片与聚类中心距离 D_next
#             next_state = torch.tensor(D_next - D).unsqueeze(1).float()  # 下一时刻状态 S'
#             next_Qvalue = Q(next_state)                                 # 根据Q网络获得下一时刻的Q值
#             # if i == iterations - 1:                                     
#             #     Q_computation = torch.tensor(reward).float()
#             # else:
#             Q_computation = torch.tensor(reward).float() + gamma * next_Qvalue   # 根据TD误差得到目标Q

#             optimizer.zero_grad()
#             loss = criterion(q_values, Q_computation)                    # Q网络更新
#             loss.backward()
#             optimizer.step()
#             print('loss:', loss)
#             writer.add_scalar('Train'+str(st) +'/Loss', loss.data, i)
#             state = next_state                                            # 状态更新
#             imgs = rotation_imgs                                          # 图片更新
#         # validation    
#         for val_x, val_y in validation_dataloader: 
#             D, D2Label,_ = genenrate_D(val_x, val_y)                       # 验证数据集
#             D1, D2Label,_ = genenrate_D(val_x, val_y)
#             state = torch.tensor(D1 - D).unsqueeze(1).float()
#             for j in range(2):
#                 q_values = Q(state)
#                 action = torch.argmax(q_values,dim=1)            
#                 rotation_imgs= step(val_x, action)
#                 D_next, _ ,_= genenrate_D(rotation_imgs, val_y)
#                 next_state = torch.tensor(D_next - D).unsqueeze(1).float()
#                 state = next_state
#                 val_x = rotation_imgs
#             _, pre_y, acc = genenrate_D(rotation_imgs,val_y)

#             Accuracy.append(acc)
#             print('the accuracy of validation is {}'.format(np.array(Accuracy).mean()))
#         writer.add_scalar('Test/Accuracy', np.array(Accuracy).mean(), st)

# torch.save(Q.state_dict(),'Q_network_600_act5.pth')
# print('Finish Q_network_600_act5 trainig---------------------')
# test


Test_acc = []
print('start tesing----------------------')
for i,(test_imgs, test_labels) in enumerate(test_dataloader): 
    # D, D2Label,_ = genenrate_D(test_imgs,test_labels)
    # D1, D2Label,_ = genenrate_D(test_imgs,test_labels)
    # state = torch.tensor(D1 - D).unsqueeze(1).float()
    # test_imgs_orignal = test_imgs
    # # 筛选旋转角
    # action_all = None
    # for j in range(5):
    #     q_values = Q(state)
    #     action = torch.argmax(q_values,dim=1)   
    #     rotation_imgs= step(test_imgs, action)
    #     if action_all is None:
    #         action_all = action.numpy()
    #         action_all = np.where(action_all==0, 10, action_all)
    #         action_all = np.where(action_all==1, 15, action_all)
    #         action_all = np.where(action_all==2, -10, action_all)
    #         action_all = np.where(action_all==3, -15, action_all)
    #         action_all = np.where(action_all==4, 0, action_all)
    #     else:
    #         action = action.numpy()
    #         action = np.where(action==0, 10, action)
    #         action = np.where(action==1, 15, action)
    #         action = np.where(action==2, -10, action)
    #         action = np.where(action==3, -15, action)
    #         action = np.where(action==4, 0, action)
    #         action_all += action
    #     D_next, _ ,_= genenrate_D(rotation_imgs, test_labels)
    #     next_state = torch.tensor(D_next - D).unsqueeze(1).float()
    #     state = next_state
    #     test_imgs = rotation_imgs

    # print("range5:", action_all.sum())
    
    # _, pre_y, acc = genenrate_D(rotation_imgs,test_labels)
    # _, pre_y_noq, acc_noq = genenrate_D(test_imgs,test_labels)
    # for p, k in enumerate(test_imgs):
    #     # 画图
    #     k = k.numpy().transpose(1,2,0)
    #     #绘制文字信息
    #     degree = action_all[p]
    #     # if pre_y[p] != test_labels[p]:
    #     if pre_y[p] != pre_y_noq[p] and pre_y[p] == test_labels[p] :
    #         plt.subplot(1, 2, 2)
    #         plt.imshow(k)
    #         plt.title("rotation" + str(degree) + "_label" + str(pre_y[p].item()))
    #         # plt.title('misclassfication' + str(pre_y[p].item()))
    #         plt.subplot(1, 2, 1)
    #         plt.imshow(test_imgs_orignal[p].numpy().transpose(1,2,0))
    #         plt.title("_label" + str(test_labels[p].item()))
    #         plt.savefig('/home/user/liuhongxing/Mnist_RL/picture_q_vs_noq/picture_' + str(p) + '.jpg')
    #         # plt.savefig('/home/user/liuhongxing/Mnist_RL/picture_failed_classification/picture_' + str(p) + '.jpg')

    _, pre_y_noq, acc = genenrate_D(test_imgs,test_labels)  # without Q
    print('the accuracy of {} batch is {}'.format(i, acc))
    # writer.add_scalar('Test/range2_Acc_with_Q',acc, i)
    Test_acc.append(acc)
print('the accuracy of test_data is {}'.format(np.array(Test_acc).mean()))           