import h5py
import numpy as np
import random
train_data=h5py.File(r'H:\model\train_catvnoncat.h5','r')
test_data=h5py.File(r'H:\model\test_catvnoncat.h5','r')#又是中文惹的事
#h5：字典
train_data.keys()#看不了，是特定格式
for key in train_data.keys():
    print(key)#类，图，标签
print(train_data['train_set_x'].shape )#209,64,64,3  209指图片数
for key in test_data.keys():
    print(key)  # 类，图，标签
print(test_data['test_set_x'].shape)
    # print(train_data['train_set_x'][:1])
#取数据
train_data_org=train_data['train_set_x'][:]
train_labels_org=train_data['train_set_y'][:]
test_data_org=test_data['test_set_x'][:]
test_labels_org=test_data['test_set_y'][:]

import matplotlib.pyplot as plt
#plt.imshow(test_data_org[148])
# data = np.random.random([4,3,3])
# print(data)
# print('#'*40)
# print(data[:1])
#拿到样本数
m_train=train_data_org.shape[0] #209
m_test=test_data_org.shape[0]#50

train_data_tran=train_data_org.reshape(m_train,-1).T
test_data_tran=test_data_org.reshape(m_test,-1).T
print(train_data_tran.shape,test_data_tran.shape)#（特征数，样本数） (12288 209)  (12288,50)
#法二
train_labels_tran=train_labels_org[np.newaxis,:]#原来50行，变成1行50列
test_labels_tran=test_labels_org[np.newaxis,:]
print(test_labels_tran.shape)#(1,209)神奇！
print(test_data_tran[:9,:9])
#数据标准化
train_data_sta=train_data_tran/255
test_data_sta=test_data_tran/255
# train_data_sta=train_data_tran-min/(max-min)

#定义sigmoid函数
def sigmoid(z):
    a=1/(1+np.exp(-z))
    return a
#初始化参数
n_dim=train_data_sta.shape[0]#取行数
w=np.zeros((n_dim,1))
b=0



def propagate(w,b,X,y):
    #前向传播函数
    z=np.dot(w.T,X)+b
    A=sigmoid(z)
    #代价函数
    m=X.shape[1]#列是样本数，行是特征数
    J=-1/m*np.sum(y*np.log(A)+(1-y)*np.log(1-A))
    #梯度下降函数
    dw=1/m*np.dot(X,(A-y).T)
    db=1/m*np.sum(A-y)
    grands={'dw':dw,'db':db}
    return grands,J

def optimizer(w,b,X,y,alpha,n_iters):
    costs=[]
    for i in range(n_iters-1):
        grands,J=propagate(w,b,X,y)
        dw=grands['dw']
        db = grands['db']
        w=w-alpha*dw
        b = b - alpha * db

        if i%100==0:
            #因为之后要画代价函数的曲线，所以把其放进List
            costs.append(J)
            print('n_iters is',i,'costs is',J)
    grands = {'dw': dw, 'db': db}
    params={'w': w, 'b': b}
    return grands,params,costs

def predict(w,b,X_test):
    #预测是用w,b进行
    z = np.dot(w.T, X_test) + b
    A=sigmoid(z)
    #设阈值
    m=X_test.shape[1]
    y_pred=np.zeros((1,m))

    for i in range(m):
        # (特征数，样本数)
        if A[:,i]>0.5:
            y_pred[:,i]=1
        else:
            y_pred[:, i] = 0
    return y_pred

#模型整合
def model(w,b,X_train,y_train,X_test,y_test,alpha,n_iters):
    grands, params, costs=optimizer(w, b, X_train, y_train, alpha, n_iters)
    w=params['w']
    b = params['b']
    y_pred_train=predict(w,b,X_train)
    y_pred_test = predict(w, b, X_test)

    print('the train acc is',np.mean(y_pred_train==y_train)*100,'%')
    print('the test acc is', np.mean(y_pred_test == y_test)*100,'%')
    b={
        'w' : w,
        'b' : b,
        'costs':costs,
        'y_pred_train' : y_pred_train,
        'y_pred_test' : y_pred_test,
        'alpha' : alpha
    }
    return b
b=model(w,b,train_data_sta,train_labels_tran,test_data_sta,test_labels_tran,alpha=0.05,n_iters=2000)
plt.plot(b['costs'])
plt.xlabel('per 100 iters')
plt.ylabel('cost')

index=1#取第2个测试样本，看真实值和预测值是否一样
print('y is',test_labels_tran[0,index])#真实值
print('y_pred is',int(b['y_pred_test'][0,index]))#预测值
plt.imshow(test_data_org[index])


