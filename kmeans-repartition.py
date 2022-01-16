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
sys.path.append("/home/xxsh/xuyaoheng/my/spark-aknn")
from utils import *
from datasets import *

datapath="/home/xxsh/xuyaoheng/my/sift/"
traindatapath=datapath+"sift_base.fvecs"
querydatapath=datapath+"sift_query.fvecs"
querygroundtruthpath=datapath+"sift_groundtruth.ivecs"


datapath="/home/xxsh/xuyaoheng/my/siftsmall/"
traindatapath=datapath+"siftsmall_base.fvecs"
querydatapath=datapath+"siftsmall_query.fvecs"
querygroundtruthpath=datapath+"siftsmall_groundtruth.ivecs"

def evaluatePredict(predict,groundtruth,k):
    l = predict.shape[0]
    real = groundtruth[:l,:k]
    cnt = 0 
    a = 0
    for i in range(l):
        len1=len(set(predict[i])&set(real[i]))
        cnt+=len1
        if len1 != k and a!=5:
            print("predict[i]\n",predict[i],"\n","real[i]\n",real[i],"\n")
            a+=1
    recall = float(cnt/(l*k))
    print("recall = cnt/float(l*k)",cnt," ","l",l,"k",k)
    return recall

def kmeansPandasDf(data,k=8,traindatanum=2000):
    traindata=data[0:traindatanum]
    l = len(data)
    df=pd.DataFrame(data) #(np.arange(l),columns=['id'])
    df['id']=np.arange(l)
    df['features']=data.tolist()
    kmeans = km(n_clusters=k, random_state=0).fit(traindata)
    res = kmeans.predict(data).reshape(l,1).tolist()
    res=list(_flatten(res))
    df["partition"] = res

    centroids = kmeans.cluster_centers_
    return df,centroids

def kmeansPandasDfV2(data,k1=8,k2=30,traindatanum=2000):
    traindata=data[0:traindatanum]
    l = len(data)
    df=pd.DataFrame(columns=['id','features','partition'])#(np.arange(l),columns=['id'])
    df['id']=np.arange(l)
    df['features']=data.tolist()
    kmeans2 = km(n_clusters=k2, random_state=0).fit(traindata)
    res = kmeans2.predict(data).reshape(l,1).tolist()
    res=list(_flatten(res))
    df["partition"] = res

    kmeans1 = km(n_clusters=k1, random_state=0).fit(traindata)
    centroids2 = kmeans2.cluster_centers_
    centroids1 = kmeans1.cluster_centers_
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
def cal(centroids1,centroids2,k1,k2):
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
def cal1(df):
    dict_of_df = {k: v for k, v in df.groupby('partition')}
    eachpartitonnum=[]
    for i in range(len(dict_of_df)):
        l=len(dict_of_df[i])
        eachpartitonnum.append(l)
    return eachpartitonnum


#k1 8 k2 30
def cal2(allpartitionrank,eachpartitonnum,k1,k2,datacnt):
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
    p.init_index(max_elements=max_elements, ef_construction=100, M=16)
    p.set_ef(10)
    p.set_num_threads(4)  # by default using all available cores
    p.add_items(data)
    return p

def processQueryVecv2(model,queryVec,globaIndexDf,queryPartitionsCol,partionmap,partitionnum=8,topkPartitionNum=3,knnQueryNum=30):
    labels, distances = model.knn_query(queryVec, k=knnQueryNum)
    cols = getMapCols(globaIndexDf,labels,queryPartitionsCol)
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
    return cur

#df column id features partition
def getsampledata(df,samplerate=0.05):
    sampledf=pd.DataFrame(columns=['id','features','partition'])
    groups=df.groupby('partition')
    for name, group in groups:
        tmp=group.sample(frac=0.05)
        sampledf=pd.concat([sampledf,tmp],axis=0)
    return sampledf


"""
    for i in range(k1):
        partitionrank = allpartitionrank[i]
        curnum = curnumlist[i]
        tmp=[]
        for j in range(k2):
            partititonid=partitionrank[j]
            if(curnum >= thresold):
                break
            if(flag[partititonid]==1):
                 continue
            tmp.append(partititonid)
            flag[partititonid]=1
            curnum += eachpartitonnum[partititonid]
        
        repartitionres.append(tmp)
        repartitionnum.append(curnum)
    assert np.sum(flag)==k2
    return repartitionres,repartitionnum

"""
# 步骤1
groundtruth = ivecs_read(querygroundtruthpath)
numpyquerydata = fvecs_read(querydatapath)
data = fvecs_read(traindatapath)

datalen=len(data)
k1=8
k2=80

df,centroids1,centroids2=kmeansPandasDfV2(data,k1=k1,k2=k2,traindatanum=int(datalen*0.1))

# 步骤2
allpartitionrank=cal(centroids1,centroids2,k1,k2)

print("allpartitionrank\n",allpartitionrank)
eachpartitonnum=cal1(df)
print("df.shape[0]",df.shape[0])
repartitionres,repartitionnum=cal2(allpartitionrank,eachpartitonnum,k1,k2,df.shape[0])
cur = np.array(repartitionnum)/df.shape[0]
print("repartitionres",repartitionres)
print("repartitionnum",repartitionnum,"repartitionnumsum",np.sum(repartitionnum))
print(cur)







sampledata_df=getsampledata(df)
"""
allldd = pd.DataFrame(sampledata)
globaldata=sampledata[0]
for i in range(1,len(sampledata)):
    globaldata+=sampledata[i]

globaldata=np.array(globaldata)
"""
"""
cur=list(sampledata[0])
for i in range(1,len(sampledata)):
    cur=np.vstack((cur,np.array(sampledata[i])))
"""



globaldata=sampledata_df["features"].to_numpy().tolist()
globaldata=np.array(globaldata,dtype=int)
model = hnsw_global_index(globaldata,1000000,128)

# processQueryVecv2(model,queryVec,globaIndexDf,queryPartitionsCol,partionmap,partitionnum=8,topkPartitionNum=3,knnQueryNum=10)

queryPartitionsCol='partition'
partitionnum=8
partionmap=getrepartitionmap(repartitionres,k1,k2)
queryvec = processQueryVecv2(model,numpyquerydata,\
    sampledata_df,queryPartitionsCol,partionmap,partitionnum=partitionnum,\
        topkPartitionNum=3,knnQueryNum=10)





















dict_of_df = {k: v for k, v in df.groupby('partition')}
res=[]
ratio=[]
for i in range(k2):
    l=len(dict_of_df[i])
    res.append(l)
print("res num",len(res))
print(res)


res.sort()
for i in range(k2):
    ratio.append(res[i]/datalen)
print(res)
print(ratio)
print(max(ratio),min(ratio))

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

"""
k = 100
querynum = 200
partitionnum=[4,8,16,32]
for i in range(4):
    func(k,querynum,partitionnum[i])
    print("done")


# 0.05
[83182, 91818, 114801, 116758, 123394, 130667, 159411, 179969]
[0.083182, 0.091818, 0.114801, 0.116758, 0.123394, 0.130667, 0.159411, 0.179969]

# 0.1
[73461, 96063, 116966, 122689, 127867, 135290, 146821, 180843]
[0.073461, 0.096063, 0.116966, 0.122689, 0.127867, 0.13529, 0.146821, 0.180843]


# 100分区
[9611, 10225, 10631, 10744, 10903, 10999, 11897, 12584, 13011, 13029, 13259, 14557, 14941, 16363, 16600, 17654, 18002, 18359, 19256, 19370, 19421, 19547, 20558, 20759, 21273, 21590, 21638, 21734, 21896, 21963, 22587, 22636, 22908, 23122, 23155, 23459, 23720, 23730, 24031, 24555, 24620, 24958, 25462, 26472, 26518, 27022, 27820, 28266, 28844, 33741]
[0.009611, 0.010225, 0.010631, 0.010744, 0.010903, 0.010999, 0.011897, 0.012584, 0.013011, 0.013029, 0.013259, 0.014557, 0.014941, 0.016363, 0.0166, 0.017654, 0.018002, 0.018359, 0.019256, 0.01937, 0.019421, 0.019547, 0.020558, 0.020759, 0.021273, 0.02159, 0.021638, 0.021734, 0.021896, 0.021963, 0.022587, 0.022636, 0.022908, 0.023122, 0.023155, 0.023459, 0.02372, 0.02373, 0.024031, 0.024555, 0.02462, 0.024958, 0.025462, 0.026472, 0.026518, 0.027022, 0.02782, 0.028266, 0.028844, 0.033741]
0.033741 0.009611
"""