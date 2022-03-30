'''
分类模型--label对应的为
1。模型的初始化
2。模型的损失
3。损失的优化
'''
import numpy as np
import torch
import cswin
from timm import create_model
import torch.nn.functional as F
import torch.optim as optim
import argparse
from get_loader import get_loaders
'''
随机初始化tensor矩阵
'''
def glorot(shape):
    """Glorot & Bengio (AISTATS 2010) init."""
    init_range = np.sqrt(6.0/(shape[0]+shape[1]))
    initial = torch.Tensor(shape[0], shape[1]).uniform_(-init_range, init_range)
    return initial
def get_loss(input,labels):
    '''

    :param input: the output of model
    :param label: the label of image
    :return: loss
    '''
    print('begin loss')
    bs = input.shape[0]
    ln=input.shape[1]
    losses=torch.zeros((1,ln)).cuda()

    for i in range(0,bs):
        get=input[i]
        label=labels[i]
        print('label is {}'.format(label.shape))
        loss=-get*label
        losses+=loss
    return torch.sum(losses)/bs


class cswin_classfier(torch.nn.Module):
    def __init__(self,label_num):
        super(cswin_classfier,self).__init__()
        self.model=create_model('CSWin_144_24322_large_224', pretrained=False, num_classes=label_num,)


        #para=torch.load('../model/cswin_large_22k_224.pth',map_location='cpu')['state_dict_ema']
        self.weight = torch.nn.Parameter(glorot([label_num,para[keys[-3]].shape[0]]).cuda()).cuda()  # 最后一层的权重
        # self.weight = torch.nn.Parameter(glorot([label_num,para[keys[-3]].shape[0]]).cuda()).cuda()  # 最后一层的权重
        #self.bias = torch.nn.Parameter(glorot([label_num]).cuda()).cuda()#
        self.bias = torch.nn.Parameter(torch.randn((label_num)).cuda()).cuda()#
        #self.para[keys[-2]]=self.weight
        #self.para[keys[-1]]=self.bias
        #self.model.load_state_dict(self.para)
        print('initial true')

    def forward(self,imagedata,label,para,keys):
        '''

        :param imagedata: information of image,as input of model,dimension is bs*3*224*224
        :param label: the label of image, dimension is bs
        :param para:the original model para,dimension of last is 21842
        :param keys:the original keys of model para, we need to change -2 -1 with weight and bias
        :return: loss: need to be optimized
        '''
        para[keys[-2]]=self.weight
        para[keys[-1]]=self.bias
        self.model.load_state_dict(para)
        print('load true')
        out=self.model(imagedata)#bs*101
        out=torch.nn.functional.log_softmax(out,dim=1)
        loss=get_loss(out,label)
        print('get loss')
        return loss

'''
初始化参数
'''
arg=argparse.ArgumentParser(description='cswin-classfier')
arg.add_argument('--label_num',type=int,default=51)
arg.add_argument('--learning_rate',type=float)
arg.add_argument('--weight_decay',type=float,default=5e-4)
arg.add_argument('--batch_size',type=int, default=16)
arg.add_argument('--epoch',type=int, default=20)
arg.add_argument('--data_path',type=str, default='../../hmdb51_iccv/hmdb_iccv.txt')
arg.add_argument('--label_path',type=str, default='../../hmdb51_iccv/hmdb_label.txt')
arg.add_argument('--data_label_path',type=str, default='../hmdb_iccv/hmdb_iccv_labels.txt')
Flage=arg.parse_args()


model=cswin_classfier(Flage.label_num).cuda()#初始化分类器
para=torch.load('../model/cswin_large_22k_224.pth',map_location=lambda storage, loc: storage.cuda(2))['state_dict_ema']
keys=[]
for x in para.keys():
    keys.append(x)
model.eval()
# labels=[int(i.strip()) for i in open(Flage.label_path).readlines()]
# labels=torch.Tensor(labels).cuda()
optimizer=optim.Adam([model.bias,model.weight],lr=Flage.learning_rate,weight_decay=Flage.weight_decay)
datas=get_loaders(Flage.data_label_path,Flage.batch_size)
for epoch in range(Flage.epoch):
    model.train()
    count=0
    for index,ind_data_i in enumerate(datas):
        ind_data=ind_data_i[0].cuda()
        trueid=ind_data_i[-1].cuda()
        loss=model(ind_data,trueid,para,keys)
        print('para {}'.format(para[keys[-1]].shape))
        print("Epoch:", '%04d' % (epoch + 1),"sample_batch:", '%04d' % (count + 1), "train_loss=", "{:.5f}".format(loss.item()))
        loss.backward()
        optimizer.step()
        count+=1

    torch.save(model.state_dict(), '../getmodel/cswin-classfier{}.pth'.format(epoch))
print('training end')
# torch.save(model.state_dict(),'cswin-classfier.pth')


