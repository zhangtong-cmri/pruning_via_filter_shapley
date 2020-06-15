import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import math


def feature_extract(feature):
    m = nn.AdaptiveAvgPool2d((1,1))
    for i in range(feature.size()[1]):
        for j in range (feature.size()[0]):
            feature[j][i] = (feature[j][i] -feature[j][i].min())/(feature[j][i].max()-feature[j][i].min())
            if j==0:
                a = feature[j][i]
            else:
                a = torch.cat((a,feature[j][i]),0)
        a= a.view(feature.size()[0],feature.size()[2],feature.size()[3])
        a= m(a)
        a = a.view(feature.size()[0],1)
        if i == 0:
            b = a
        else:
            b = torch.cat((b,a),1)
    b = b.view(feature.size()[1],feature.size()[0])
    return b 

def entropy_extract(feature):
    tmp=[0]*256
    val=0
    k=0
    res=0
    for i in range(len(feature)):
        val = feature[i]
        tmp[val] = float(tmp[val] + 1)
        k =  float(k + 1)
    for i in range(len(tmp)):
        tmp[i] = float(tmp[i] / k)
    for i in range(len(tmp)):
        if(tmp[i] == 0):
            res = res
        else:
            res = float(res - tmp[i] * (math.log(tmp[i]) / math.log(2.0)))
    return res


def extract_subconv_entropy_list(feature):
   
    tmp=[]
    for i in range (feature.size()[0]):
        c= feature[i,:].data.cpu().numpy()
        c= np.asarray(c * 255, dtype=np.uint8)
        entropy = entropy_extract(c)
        tmp.append(entropy)
    return tmp


def extract_Accumulation_entropy_list(feature):
    
    """
        feature1=feature_extract(feature)
        feature2 =extract_Accumulation_entropy_list(feature1)
    """
    tmp =[]
    for i in range(feature.size()[0]):
        c= feature[i,:].data.cpu().numpy()
        c= np.asarray(c * 255, dtype=np.uint8)
        if i!=0:
            d = np.hstack((d,c))
            entropy = entropy_extract(d)
            tmp.append(entropy)
        else:
            entropy = entropy_extract(c)
            d =c
            tmp.append(entropy)
    return tmp

def filter_shapely(feature):
    """
         feature1 = feature_extract(feature)
         feature2 = filter_shapely(feature1)
    """
    tmp = extract_Accumulation_entropy_list(feature)
    tmp2=[]
    for i in range(len(tmp)):
        if i!=0:
            tmp2.append(tmp[i]-tmp[i-1])
        else:
            tmp2.append(tmp[i])
    return tmp2

def get_small_value_filter_shapely(feature,compress_rate):
    
    """
        feature1=feature_extract(feature)
        shapely =filter_shapely(feature1)
        get_small_value_filter_shapely(shapely,0.5)
    """
    feature =np.array(feature)
    filter_shapely_index =feature.argsort()[:int(len(feature)*compress_rate)]
    return filter_shapely_index



def get_codebook(feature):
    
    """
        feature1 = feature_extract(feature)
        shapely = filter_shapely(feature1)
        codebook_index = get_codebook(shapely)
    """
    codebook =[]
    for i in range(len(feature)):
        if feature[i]<0:
            codebook.append(i) 
    return codebook

def randomized_policy_filter_shapely(feature,time=5):
    """
         feature1 = feature_extract(feature)
         shapely = randomized_policy_filter_shapely(feature1)
    """
    b=[0]*(feature.size()[0])
    for a in range (time): 
        index = [i for i in range(feature1.size()[0])]
        np.random.shuffle(index)
        a=[]
        for j in range (feature1.size()[0]):
            feature4 = filter_shapely(feature1)
            for i in range (feature1.size()[0]):
                feature1[i] = feature1[index[i]]
                if index[i]== j:
                    a.append(i)

            b[j]=b[j]+feature4[a[0]]
    b = [i/time for i in b ]
    
    return b


def get_new_conv_out(conv, dim, channel_index, independent_prune_flag=False):
    if dim == 0:
        conv.out_channels = int(conv.out_channels -len(channel_index))
        conv.weight.data = index_remove(conv.weight.data, 0, channel_index)
        
def get_new_conv_in(conv, channel_index, independent_prune_flag=False):
    conv.in_channels= int(conv.in_channels - len(channel_index))
    conv.weight.data = index_remove(conv.weight.data, 1, channel_index)


def get_new_norm(norm, channel_index):
    norm.num_features=int(norm.num_features - len(channel_index))
    norm.weight.data = index_remove(norm.weight.data, 0, channel_index)
    norm.bias.data = index_remove(norm.bias.data, 0, channel_index)

    if norm.track_running_stats:
        norm.running_mean.data = index_remove(norm.running_mean.data, 0, channel_index)
        norm.running_var.data = index_remove(norm.running_var.data, 0, channel_index)

def get_new_linear(linear, channel_index):
    linear.in_features = int(linear.in_features-len(channel_index))
    linear.weight.data = index_remove(linear.weight.data, 1, channel_index)  

def index_remove(tensor, dim, index, removed=False):
    if tensor.is_cuda:
        tensor = tensor.cpu()
        
    size_ = list(tensor.size())
    new_size = tensor.size(dim) - len(index)
    size_[dim] = new_size
    new_size = size_
    select_index = list(set(range(tensor.size(dim))) - set(index))
    new_tensor = torch.index_select(tensor, dim, torch.tensor(select_index))
#     print("new_tensor.shape:",new_tensor.shape)
    
    if removed:
        return new_tensor, torch.index_select(tensor, dim, torch.tensor(index))

    return new_tensor

