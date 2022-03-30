'''
1.create model -> load model -> load data of model
2.video data: video txt(the image from video path) + ground true
3.get result -> statistics of class
4.get accu
5.
'''

from timm import create_model
import cswin
import torch
from get_loader import get_loaders
import torch.nn.functional as F
import numpy as np
import pickle

split_test='ind.hmdb51.split_test'
datatxt_id=[i.strip().split() for i in open('hmdb51_txt_label.txt').readlines()]
def get_testdata(split_test,datatxt_id):
    data=pickle.load(open(split_test,'rb'))# 50*25
    test_mask_s={}
    test_ind_s={}
    for split_id in range(data.shape[0]):
        zero_test_classes=data[split_id]-1
        ind_test=[]
        for i in range(len(datatxt_id)):
            if int(datatxt_id[i][1]) in zero_test_classes:
                ind_test.append(i)
        mask=np.zeros(51)
        mask[zero_test_classes]=1
        test_mask=np.array(mask, dtype=np.bool)
        test_mask_s[split_id]=test_mask
        test_ind_s[split_id]=ind_test
    return test_mask_s,test_ind_s

'''
1.加载模型
'''
testmodel=create_model('CSWin_144_24322_large_224', pretrained=False, num_classes=51,)
#para=torch.load('../model/cswin_large_22k_224.pth',map_location='cpu')['state_dict_ema']
para=torch.load('../getmodel/cswin-classfier0.pth',map_location='cpu')
#key=[x for x in para.keys()]
#key1=[x for x in para1.keys()]
#para[key[-2]]=para1[key1[0]]
#para[key[-1]]=para1[key1[1]]
testmodel.load_state_dict(para)
testmodel=testmodel.eval()
testmodel.cuda()
print('load model is true')
test_mask_s,test_id_s=get_testdata(split_test,datatxt_id)
print('get data is true')
accu=[]
f=open('result/accu.txt','a')
for i in range(0,1):
    print('split is {}'.format(i))
    test_mask=torch.Tensor(np.array(test_mask_s[i])).cuda()
    test_id=np.array(test_id_s[i])
    testcount=len(test_id)
    truecount=0
    for x in test_id:
        videotxt=datatxt_id[x][0]
        trueid=torch.Tensor(datatxt_id[x][1])
        test_loader = get_loaders(videotxt)
        for batch_idx, data in enumerate(test_loader):
            with torch.no_grad():
                img_t = data[0].cuda()
                out = testmodel(img_t)
                out = out.data.numpy()
                if batch_idx == 0:
                    # prediction = pred
                    out_tensor = out
                else:
                    out_tensor = np.r_[out_tensor, out]
        out_tensor=torch.Tensor(out_tensor)#len(image)*51
        out_tensor=out_tensor*test_mask
        out_tensor=F.softmax(out_tensor,dim=1)
        out_tensor=torch.argmax(out_tensor,dim=1)#len(image)
        prectid=torch.mode(out_tensor)[0]#求一组数据的众数
        if prectid==trueid:truecount+=1
    split_accu=truecount/testcount
    print('split is {},accu is {}'.format(i,split_accu))
    f.write('split is {},accu is {}'.format(i,split_accu)+'\n')
    accu.append(split_accu)
f.close()





