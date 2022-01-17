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
from kmeans_repartition_utils import *
datapath="/home/xxsh/xuyaoheng/my/sift/"
traindatapath=datapath+"sift_base.fvecs"
querydatapath=datapath+"sift_query.fvecs"
querygroundtruthpath=datapath+"sift_groundtruth.ivecs"


datapath="/home/xxsh/xuyaoheng/my/siftsmall/"
traindatapath=datapath+"siftsmall_base.fvecs"
querydatapath=datapath+"siftsmall_query.fvecs"
querygroundtruthpath=datapath+"siftsmall_groundtruth.ivecs"

# 步骤1
groundtruth = ivecs_read(querygroundtruthpath)
numpyquerydata = fvecs_read(querydatapath)
data = fvecs_read(traindatapath)

datalen=len(data)
k1=8
k2=80

df,centroids1,centroids2=kmeansPandasDfV2(data,k1=k1,k2=k2,traindatanum=int(datalen*0.1))

# 步骤2
allpartitionrank=getallpartitionrank(centroids1,centroids2,k1,k2)

print("allpartitionrank\n",allpartitionrank)
eachpartitonnum=geteachpartitonnum(df)
print("df.shape[0]",df.shape[0])
repartitionres,repartitionnum=repartition(allpartitionrank,eachpartitonnum,k1,k2,df.shape[0])
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





"""
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