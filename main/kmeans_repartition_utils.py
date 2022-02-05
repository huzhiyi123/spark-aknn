from re import S
import sys
import pandas as pd

from tempfile import gettempdir
import pandas as pd
import numpy as np
from tkinter import _flatten
import pickle
from sklearn.cluster import KMeans as km
import sys
import hnswlib
sys.path.append("..")
from utils import *
from datasets import *

def kmeansPandasDfV1(data,k1=8,traindatanum=2000,partitioncolname="partition"):
    traindata=data[0:traindatanum]
    l = len(data)
    df=pd.DataFrame(columns=['id','features',partitioncolname])#(np.arange(l),columns=['id'])
    df['id']=np.arange(l)
    df['features']=data.tolist()
    kmeans2 = km(n_clusters=k1, random_state=0).fit(traindata)
    res = kmeans2.predict(data).reshape(l,1).tolist()
    res=list(_flatten(res))
    df[partitioncolname] = res
    return df

def kmeansPandasDfV2(data,k1=8,k2=30,traindatanum=2000,partitioncolname="partition"):
    traindata=data[0:traindatanum]
    l = len(data)
    df=pd.DataFrame(columns=['id','features',partitioncolname])#(np.arange(l),columns=['id'])
    df['id']=np.arange(l)
    df['features']=data.tolist()
    kmeans2 = km(n_clusters=k2, random_state=0).fit(traindata)
    res = kmeans2.predict(data).reshape(l,1).tolist()
    res=list(_flatten(res))
    df[partitioncolname] = res
    kmeans1 = km(n_clusters=k1, random_state=0).fit(traindata)
    centroids2 = kmeans2.cluster_centers_
    centroids1 = kmeans1.cluster_centers_
    return df,centroids1,centroids2



def kmeansPandasDfV3(data,partitioncsvpath,centroids1path,centroids2path,partitioncolname="partition"):
    l = len(data)
    df=pd.DataFrame(columns=['id','features',partitioncolname])#(np.arange(l),columns=['id'])
    df['id']=np.arange(l)
    df['features']=data.tolist()
    partitioncoldata=pd.read_csv(partitioncsvpath).values.astype(np.int) 
    #print("partitioncoldata[0:10]",partitioncoldata[0:10])
    df[partitioncolname]= partitioncoldata
    centroids1 = pd.read_csv(centroids1path).values
    centroids2 = pd.read_csv(centroids2path).values
    return df,centroids1,centroids2



def getsampledata(df,samplerate=0.05):
    sampledata=[]
    groups=df.groupby('partition')
    for name, group in groups:
        sampledata.append(group["features"].sample(frac=0.05))
    return sampledata
 


"""
# 分k个分区 得到k个质心
# 按照质心与k个质心中值的位置 
# 选择远的质心 
ad=np.linalg.norm(centroids1[0]-centroids2[0])
print("ad=np.linalg.norm(centroids1[0]-centroids2[0])",ad)
res=np.zeros((k1,k2))
for i in range(k1):
    vec1 = centroids1[i]
    for j in range(k2):
        vec2 = centroids2[j]
        dist = np.linalg.norm(vec1-vec2)
        res[i][j] = dist

partion=np.array(range(30))
all=[]
for i in range(k1):
    cur = res[i]
    partition = np.array(range(k2))
    c = np.lexsort((partition,cur))
    all.append(list(zip(cur[c],partition[c])))
# 根据距离对partition进行排序
# 
print(all[0])
"""
def get_cos_dist(v1: list, v2: list):
    num = float(np.dot(v1, v2))  # 向量点乘
    denom = np.linalg.norm(v1) * np.linalg.norm(v2)  # 求模长的乘积
    dist=1-num/denom
    return dist
# 计算各个质心最近分区 按距离排序
def getallpartitionrank(centroids1,centroids2,k1,k2):
    res=np.zeros((k1,k2))
    for i in range(k1):
        vec1 = centroids1[i]
        for j in range(k2):
            vec2 = centroids2[j]
            dist = get_cos_dist(vec1,vec2) #np.linalg.norm(vec1-vec2)
            res[i][j] = dist
    allpartitionrank = np.zeros((k1,k2),int)
    for i in range(k1):
        cur = res[i]
        partition = np.array(range(k2))
        c = np.lexsort((partition,cur))
        allpartitionrank[i]=partition[c]
    return allpartitionrank

# 计算各个分区数量
def geteachpartitonnum(df):
    dict_of_df = {k: v for k, v in df.groupby('partition')}
    eachpartitonnum=[]
    for i in range(len(dict_of_df)):
        l=len(dict_of_df[i])
        eachpartitonnum.append(l)
    return eachpartitonnum


#k1 8 k2 30
def repartition(allpartitionrank,eachpartitonnum,k1,k2,datacnt):
    thresold=int((datacnt/k1))
    ceil = thresold
    repartitionres=[]
    repartitionnum=np.zeros((k1,1),int)
    for i in range(k1):
        repartitionres.append([])
    flag=np.zeros((k2,1),int)

    while(np.sum(flag) != k2):
        for i in range(k1):
            partitionrank = allpartitionrank[i]
            curnum = repartitionnum[i]
            if(curnum >= thresold):
                continue
            for j in range(k2):
                partititonid=partitionrank[j]
                if(flag[partititonid]==1):
                    continue
                flag[partititonid]=1
                repartitionnum[i] += eachpartitonnum[partititonid]
                repartitionres[i].append(partititonid)
                break
    return repartitionres,repartitionnum


def getrepartitionmap(repartitionres,k1,k2):
    res = np.zeros(k2,int)
    for i in range(k1):
        cur = repartitionres[i]
        for j in cur:
            res[j]=i
    return res


def hnsw_global_index(data,max_elements,dim):
    p = hnswlib.Index(space='cosine', dim=dim)  # possible options are l2, cosine or ip
    p.init_index(max_elements=max_elements, ef_construction=100, M=32)
    p.set_ef(30)
    p.set_num_threads(4)  # by default using all available cores
    p.add_items(data)
    return p

def hnsw_global_index_wrapper(sampledata_df,max_elements,dim,featurecol='features'):
    globaldata=sampledata_df["features"].to_numpy().tolist()
    data=np.array(globaldata,dtype=int)
    p = hnswlib.Index(space='cosine', dim=dim)  # possible options are l2, cosine or ip
    p.init_index(max_elements=max_elements, ef_construction=100, M=16)
    p.set_ef(10)
    p.set_num_threads(4)  # by default using all available cores
    p.add_items(data)
    return p


def processQueryVecv2(model,queryVec,globaIndexDf,queryPartitionsCol,\
    partitionCol,partionmap,partitionnum=8,topkPartitionNum=3,knnQueryNum=30):
    # 找到最近的几个向量的位置
    T6 = time.time()
    labels, distances = model.knn_query(queryVec, k=knnQueryNum)
    T7 = time.time()
    globalindexconstructtime=(T7-T6)*1000

    #print("processQueryVecv2 globaIndexDf.shape ",globaIndexDf.shape)
    cols = getMapCols(globaIndexDf,labels,partitionCol)
    # unique 这些分区号 不足的填充其他分区 返回的是list
    for i in range(len(cols)):
        for j in range(knnQueryNum):
            tmp=cols[i][j]
            cols[i][j]=partionmap[tmp]
    # 增加根据cols找到 真正分区 
    cols = uniqueAndRefill(np.array(cols),topkPartitionNum,partitionnum)
    length = queryVec.shape[0]
    cur = pd.DataFrame(np.arange(length),columns=["id"])
    cur['features'] = queryVec.tolist()
    cur[queryPartitionsCol] = cols
    return cur,globalindexconstructtime

#df column id features partition
def getsampledata(df,samplerate=0.05):
    sampledf=pd.DataFrame(columns=['id','features','partition'])
    groups=df.groupby('partition')
    for name, group in groups:
        tmp=group.sample(frac=0.05)
        sampledf=pd.concat([sampledf,tmp],axis=0)
    return sampledf


## 分区后各个分区占比
def func(k,querynum,partitionnum,traindatanum=2000):
    print("k,querynum,partitionnum",k,querynum,partitionnum)
    df=kmeansPandasDf(data,k=partitionnum,traindatanum=2000)
    dict_of_df = {k: v for k, v in df.groupby('partition')}

    res= []
    for i in range(partitionnum):
        tmp=dict_of_df[i]['id'].values.tolist()
        res.append(tmp)

    real = groundtruth[:querynum,:k]
    my = np.zeros(partitionnum*querynum).reshape(querynum,-1)
    for i in range(querynum):
        for j in range(partitionnum):
            cur = len(set(real[i])&set(res[j]))
            my[i][j]=cur
        #cnt+=len(set(predict[i])&set(real[i]))
    my=np.sort(my)
    #print(my)
    c=np.sum(my,axis=0)/(querynum*k)
    fg=np.count_nonzero(my,axis=0)
    kg=np.sum(my, axis=0)
    print("fg=np.count_nonzero(my,axis=0)\n")
    print(fg)
    print("kg=np.sum(my, axis=0)\n")
    print(kg)
    print("c=np.sum(my,axis=0)/(querynum*k)\n")
    print(c)
