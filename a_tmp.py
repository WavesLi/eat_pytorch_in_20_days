import datetime
import random
import pandas as pd
import math
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset,DataLoader,TensorDataset
from sklearn.metrics import accuracy_score
import plotly.graph_objects as go
import plotly.offline as of


train_data = pd.read_csv("./data/titanic/train.csv")
test_data = pd.read_csv('./data/titanic/test.csv')
print(train_data.columns)
print(train_data.index)
print(train_data.info())
print(train_data)

def DataClean(train_data):
    df_result = pd.DataFrame()
    df_Pclass = pd.get_dummies(train_data["Pclass"])
    df_Pclass.columns = ["PC_1","PC_2","PC_3"]
    df_result = pd.concat((df_result,df_Pclass),axis=1)
    df_Sex = pd.get_dummies(train_data["Sex"])
    df_result = pd.concat((df_result,df_Sex),axis=1)
    df_Age_isna = pd.DataFrame(train_data["Age"].isnull().map(lambda x:int(x)))
    df_Age_isna.columns = ["Age_Na"]
    df_Age = pd.DataFrame(train_data["Age"].fillna(0).map(lambda x:math.ceil(x)))
    df_Age.columns = ["Age"]
    df_result  = pd.concat((df_result,df_Age,df_Age_isna),axis=1)
    df_result['SibSp'] = train_data['SibSp']
    df_result['Parch'] = train_data['Parch']
    df_Fare = pd.DataFrame(train_data["Fare"].map(lambda x:math.ceil(x)))
    df_result  = pd.concat((df_result,df_Fare),axis=1)
    df_Embarked = pd.get_dummies(train_data['Embarked'],dummy_na=True)
    df_Embarked.columns = ['Embarked_' + str(x) for x in df_Embarked.columns]
    df_result = pd.concat([df_result,df_Embarked],axis = 1)
    df_Cabin_isna = train_data["Cabin"].fillna("X").map(lambda x:x[0] if x!="X" else x)
    df_Cabin_isna = pd.DataFrame(df_Cabin_isna)
    df_Cabin_isna = pd.get_dummies(df_Cabin_isna["Cabin"])
    df_result = pd.concat([df_result,df_Cabin_isna],axis = 1)
    return df_result

x_train = DataClean(train_data)
x_test  = DataClean(test_data[:176])
x_test["T"] = pd.Series([0 for i in range(176)])
x_train = x_train.values
y_train = train_data[['Survived']].values

x_test = x_test.values
y_test = test_data[['Survived']].values[:176]

print("x_train.shape =", x_train.shape)
print("x_test.shape =", x_test.shape)

print("y_train.shape =", y_train.shape)
print("y_test.shape =", y_test.shape)
dl_train = DataLoader(TensorDataset(torch.tensor(x_train).float(),torch.tensor(y_train).float()),
                     shuffle = True, batch_size = 8)
dl_valid = DataLoader(TensorDataset(torch.tensor(x_test).float(),torch.tensor(y_test).float()),
                     shuffle = False, batch_size = 8 )


# def create_net():
#     net = nn.Sequential()
#     net.add_module("linear1", nn.Linear(23, 69))
#     net.add_module("dropout1", nn.Dropout(0.2))
#     net.add_module("relu1", nn.ReLU())
#     net.add_module("linear2", nn.Linear(69, 69))
#     net.add_module("dropout2", nn.Dropout(0.2))
#     net.add_module("relu2", nn.ReLU())
#     net.add_module("linear3", nn.Linear(69, 69))
#     net.add_module("dropout3", nn.Dropout(0.2))
#     net.add_module("relu3", nn.ReLU())
#     net.add_module("linear4", nn.Linear(69, 69))
#     net.add_module("relu4", nn.ReLU())
#     net.add_module("linear5", nn.Linear(69, 69))
#     net.add_module("relu5", nn.ReLU())
#     net.add_module("linear6", nn.Linear(69, 1))
#     net.add_module("relu6", nn.ReLU())
#     net.add_module("sigmoid", nn.Sigmoid())
#     return net
#
# net = create_net()
class Net(torch.nn.Module):

    def dynamic(self):
        self.dropout1 = torch.nn.Dropout(0.15)
        self.dropout2 = torch.nn.Dropout(0.17)
        self.dropout3 = torch.nn.Dropout(0.19)
        self.dropout4 = torch.nn.Dropout(0.21)
        self.dropout5 = torch.nn.Dropout(0.22)
        self.dropout6 = torch.nn.Dropout(0.24)
        self.dropout7 = torch.nn.Dropout(0)
        self.dropout8 = torch.nn.Dropout(0.3)
        self.dropout9 = torch.nn.Dropout(0.4)
        self.dropout10 = torch.nn.Dropout(0.5)
        self.dropout11 = torch.nn.Dropout(0.6)
        self.dropout12 = torch.nn.Dropout(0.7)
        self.dropout13 = torch.nn.Dropout(0.8)
        self.dropout14= torch.nn.Dropout(0.9)

    def __init__(self):
        super(Net, self).__init__()
        self.layer1 = torch.nn.Linear(in_features=23, out_features=69)
        self.nm1 = nn.LayerNorm([69,])
        self.relu1 = nn.Tanh()
        self.layer2 = torch.nn.Linear(in_features=69, out_features=69)
        self.nm2 = nn.LayerNorm([69,])
        self.relu2 = nn.Tanh()
        self.layer3 = torch.nn.Linear(in_features=69, out_features=69)
        self.nm3 = nn.LayerNorm([69,])
        self.relu3 = nn.Tanh()
        self.layer4 = torch.nn.Linear(in_features=69, out_features=69)
        self.nm4 = nn.LayerNorm([69,])
        self.relu4 = nn.Tanh()
        self.layer5 = torch.nn.Linear(in_features=69, out_features=69)
        self.nm5 = nn.LayerNorm([69,])
        self.relu5 = nn.Tanh()
        self.layer6 = torch.nn.Linear(in_features=69, out_features=1)
        self.relu6 = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        self.dynamic()

    def forward(self, x):
        x1 = self.relu1(self.nm1(self.layer1(x)))
        x_tmp = self.dropout7(x1)
        x2 = self.relu2(self.nm2(self.layer2(x_tmp)))
        x_tmp = self.dropout7(x2)+self.dropout8(x1)
        x3 = self.relu3(self.nm3(self.layer3(x_tmp)))
        x_tmp = self.dropout7(x3)+self.dropout8(x2) + self.dropout9(x1)
        x4 = self.relu4(self.nm4(self.layer4(x_tmp)))
        x_tmp = self.dropout7(x4) +self.dropout8(x3) + self.dropout9(x2) + self.dropout10(x1)
        x5 = self.relu5(self.nm5(self.layer5(x_tmp)))
        x_tmp = self.dropout7(x5) +self.dropout8(x4) + self.dropout9(x3) + self.dropout10(x2) + self.dropout11(x1)
        x6 = self.relu6(self.layer6(x_tmp))
        y  = self.sigmoid(x6)
        return y
net = Net()

# class Net(torch.nn.Module):
#     def  dynamic(self,n):
#         self.dropout1 = torch.nn.Dropout(0.1)
#         self.dropout2 = torch.nn.Dropout(0.15)
#         self.dropout3 = torch.nn.Dropout(0.2)
#         self.dropout4 = torch.nn.Dropout(0.25)
#         self.dropout5 = torch.nn.Dropout(0.3)
#         self.dropout6 = torch.nn.Dropout(0.35)
#         self.dropout7 = torch.nn.Dropout(0.4)
#         self.dropout8 = torch.nn.Dropout(0.45)
#         self.dropout9 = torch.nn.Dropout(0.5)
#         self.dropout10 = torch.nn.Dropout(0.6)
#         self.dropout11 = torch.nn.Dropout(0.65)
#         self.dropout12 = torch.nn.Dropout(0.7)
#         self.dropout13 = torch.nn.Dropout(0.75)
#         self.dropout14= torch.nn.Dropout(0.8)
#         self.mask1_1 = torch.tensor([0 if random.random() > 0.15 else  1 for i in range( n * 69)],dtype=torch.bool).reshape((n, 69))
#         self.mask1_2 = torch.tensor([0 if random.random() > 0.15 else  1 for i in range( n * 69)],dtype=torch.bool).reshape((n, 69))
#         self.mask1_3 = torch.tensor([0 if random.random() > 0.15 else  1 for i in range( n * 69)],dtype=torch.bool).reshape((n, 69))
#         self.mask1_4 = torch.tensor([0 if random.random() > 0.15 else  1 for i in range( n * 69)],dtype=torch.bool).reshape((n, 69))
#         self.mask1_5 = torch.tensor([0 if random.random() > 0.15 else 1 for i in range( n * 69)],dtype=torch.bool).reshape((n, 69))
#         self.mask2_1 = torch.tensor([0 if random.random() > 0.25 else 1 for i in range(  n * 69)],dtype=torch.bool).reshape((n, 69))
#         self.mask2_2 = torch.tensor([0 if random.random() > 0.25 else 1 for i in range(  n * 69)],dtype=torch.bool).reshape((n, 69))
#         self.mask2_3 = torch.tensor([0 if random.random() > 0.25 else 1 for i in range(  n * 69)],dtype=torch.bool).reshape((n, 69))
#         self.mask2_4 = torch.tensor([0 if random.random() > 0.25 else 1 for i in range(  n * 69)],dtype=torch.bool).reshape((n, 69))
#         self.mask2_5 = torch.tensor([0 if random.random() > 0.25 else 1 for i in range(  n * 69)],dtype=torch.bool).reshape((n, 69))
#         self.mask3_1 = torch.tensor([0 if random.random() > 0.45 else 1 for i in range( n * 69)],dtype=torch.bool).reshape(( n,69))
#         self.mask3_2 = torch.tensor([0 if random.random() > 0.45 else  1 for i in range( n * 69)],dtype=torch.bool).reshape(( n,69))
#         self.mask3_3 = torch.tensor([0 if random.random() > 0.45 else  1 for i in range( n * 69)],dtype=torch.bool).reshape(( n,69))
#         self.mask3_4 = torch.tensor([0 if random.random() > 0.45 else  1 for i in range( n * 69)],dtype=torch.bool).reshape(( n,69))
#         self.mask3_5 = torch.tensor([0 if random.random() > 0.45 else  1 for i in range( n * 69)],dtype=torch.bool).reshape(( n,69))
#
#     def __init__(self):
#         super(Net, self).__init__()
#         self.layer1 = torch.nn.Linear(in_features=23, out_features=69)
#         self.nm1 = nn.LayerNorm([69,])
#         self.relu1 = nn.Tanh()
#         self.layer2 = torch.nn.Linear(in_features=69, out_features=23)
#         self.nm2 = nn.LayerNorm([23,])
#         self.relu2 = nn.Tanh()
#         self.layer3 = torch.nn.Linear(in_features=23, out_features=9)
#         self.nm3 = nn.LayerNorm([9,])
#         self.relu3 = nn.Tanh()
#         self.layer4 = torch.nn.Linear(in_features=9, out_features=1)
#         self.relu = nn.ReLU()
#         # self.mask1_1 = torch.tensor([1 if random.random() > 0.15 else 0 for i in range(23 * 69)],dtype=torch.bool).reshape((23, 69))
#         self.layer5 = torch.nn.Sigmoid()
#         self.dynamic(8)
#     def forward(self, x):
#         x1 = self.layer1(x)
#         x1_1 = self.dropout8(x1)
#         x1_2 = self.dropout8(x1)
#         x1_3 = self.dropout8(x1)
#         x1_4 = self.dropout8(x1)
#         x1_5 = self.dropout8(x1)
#         x2 = self.relu(x1_1+x1_2+x1_3+x1_4+x1_5)
#         # x2   = x1_1+x1_2+x1_3+x1_4+x1_5
#         x2 = self.layer2(x2)
#         x2_1 = self.dropout4(x2)
#         x2_2 = self.dropout4(x2)
#         x2_3 = self.dropout4(x2)
#         x2_4 = self.dropout4(x2)
#         x2_5 = self.dropout4(x2)
#         x3 = self.relu(x2_1+x2_2+x2_3+x2_4+x2_5)
#         # x3 = x2_1+x2_2+x2_3+x2_4+x2_5
#         x3 = self.layer3(x3)
#         x3_1 = self.dropout2(x3)
#         x3_2 = self.dropout2(x3)
#         x3_3 = self.dropout2(x3)
#         x3_4 = self.dropout2(x3)
#         x3_5 = self.dropout2(x3)
#         x4 = self.relu(x3_1+x3_2+x3_3+x3_4+x3_5)
#         # x4 = x3_1+x3_2+x3_3+x3_4+x3_5
#         x5 = self.layer4(x4)
#         y = self.layer5(x5)
#         return y
#     #
#     # def forward(self, x):
#     #     x1 = self.layer1(x)
#     #     x1_1 = self.relu1(x1.masked_fill(self.mask1_1,0))
#     #     x1_2 = self.relu1(x1.masked_fill(self.mask1_2,0))
#     #     x1_3 = self.relu1(x1.masked_fill(self.mask1_3,0))
#     #     x1_4 = self.relu1(x1.masked_fill(self.mask1_4,0))
#     #     x1_5 = self.relu1(x1.masked_fill(self.mask1_5,0))
#     #     # x2   = self.nm1(x1_1+x1_2+x1_3+x1_4+x1_5)
#     #     x2   = x1_1+x1_2+x1_3+x1_4+x1_5+0.2*x1
#     #     # x2 = x1_1+x1_2+x1_3+x1_4+x1_5
#     #     x2 = self.layer2(x2)
#     #     x2_1 = self.relu2(x2.masked_fill(self.mask2_1,0))
#     #     x2_2 = self.relu2(x2.masked_fill(self.mask2_2,0))
#     #     x2_3 = self.relu2(x2.masked_fill(self.mask2_3,0))
#     #     x2_4 = self.relu2(x2.masked_fill(self.mask2_4,0))
#     #     x2_5 = self.relu2(x2.masked_fill(self.mask2_5,0))
#     #     # x3 = self.nm2(x2_1+x2_2+x2_3+x2_4+x2_5)
#     #     x3 = x2_1+x2_2+x2_3+x2_4+x2_5+0.1*x1+0.2*x2
#     #     # x3 = x2_1+x2_2+x2_3+x2_4+x2_5
#     #     x3 = self.layer3(x3)
#     #     x3_1 = self.relu3(x3.masked_fill(self.mask3_1,0))
#     #     x3_2 = self.relu3(x3.masked_fill(self.mask3_2,0))
#     #     x3_3 = self.relu3(x3.masked_fill(self.mask3_3,0))
#     #     x3_4 = self.relu3(x3.masked_fill(self.mask3_4,0))
#     #     x3_5 = self.relu3(x3.masked_fill(self.mask3_5,0))
#     #     # x4 = self.nm3(x3_1+x3_2+x3_3+x3_4+x3_5)
#     #     x4 = x3_1+x3_2+x3_3+x3_4+x3_5+0.05*x1+0.1*x2+0.2*x3
#     #     # x4 = x3_1+x3_2+x3_3+x3_4+x3_5
#     #     x5 = self.layer4(x4)
#     #     y = self.layer5(x5)
#     #     return y
#
# net = Net()

loss_func = nn.BCELoss()
optimizer = torch.optim.Adam(params=net.parameters(),lr = 0.001,weight_decay=0.005)
metric_func = lambda y_pred,y_true: accuracy_score(y_true.data.numpy(),y_pred.data.numpy()>0.5)
# metric_func2 = lambda y_pred,y_true: accuracy_score(y_true.data.numpy(),y_pred.data.numpy()>0.6)
metric_name = "accuracy"

epochs = 2000
log_step_freq = 30

dfhistory = pd.DataFrame(columns=["epoch", "loss", metric_name, "val_loss", "val_" + metric_name])
print("Start Training...")
nowtime = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
print("==========" * 8 + "%s" % nowtime)

for epoch in range(1, epochs + 1):

    # 1，训练循环-------------------------------------------------
    net.train()
    loss_sum = 0.0
    metric_sum = 0.0
    step = 1

    for step, (features, labels) in enumerate(dl_train, 1):

        # 梯度清零
        optimizer.zero_grad()

        # 正向传播求损失
        predictions = net(features)
        loss = loss_func(predictions, labels)
        metric = metric_func(predictions, labels)

        # 反向传播求梯度
        loss.backward()
        optimizer.step()

        # 打印batch级别日志
        loss_sum += loss.item()
        metric_sum += metric.item()
        if step % log_step_freq == 0:
            print(("[step = %d] loss: %.3f, " + metric_name + ": %.3f") %
                  (step, loss_sum / step, metric_sum / step))

    # 2，验证循环-------------------------------------------------
    net.eval()
    val_loss_sum = 0.0
    val_metric_sum = 0.0
    val_step = 1

    for val_step, (features, labels) in enumerate(dl_valid, 1):
        # 关闭梯度计算
        with torch.no_grad():
            predictions = net(features)
            val_loss = loss_func(predictions, labels)
            val_metric = metric_func(predictions, labels)
            val_loss_sum += val_loss.item()
            val_metric_sum += val_metric.item()

    # 3，记录日志-------------------------------------------------
    info = (epoch, loss_sum / step, metric_sum / step,
            val_loss_sum / val_step, val_metric_sum / val_step)
    dfhistory.loc[epoch - 1] = info

    # 打印epoch级别日志
    print(("\nEPOCH = %d, loss = %.3f," + metric_name + \
           "  = %.3f, val_loss = %.3f, " + "val_" + metric_name + " = %.3f")
          % info)
    nowtime = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print("\n" + "==========" * 8 + "%s" % nowtime)

print('Finished Training...')

print(dfhistory.info)
dfhistory.to_excel("/Users/litao/Desktop/litao/github/GitHub/eat_pytorch_in_20_days/data/titanic/my_train.xlsx")
