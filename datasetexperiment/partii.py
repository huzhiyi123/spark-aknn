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


aknnfold="/home/yaoheng/test/spark-aknn/"
sys.path.append(aknnfold)
sys.path.append(aknnfold+"test")
from utils import *
from datasets import *
from kmeans_repartition_utils import *
from utils import *
from datasets import *




datapath="/home/yaoheng/test/data/siftsmall/"
traindatapath=datapath+"siftsmall_base.fvecs"
querydatapath=datapath+"siftsmall_query.fvecs"
querygroundtruthpath=datapath+"siftsmall_groundtruth.ivecs"

groundtruth = ivecs_read(querygroundtruthpath)
numpyquerydata = fvecs_read(querydatapath)
traindata = fvecs_read(traindatapath)

datalen=len(traindata)
querynum=len(numpyquerydata)

kmeanstrainrate = 0.05

partitioncolname="partition"
queryPartitionsCol='querypartitions'
nfcolname="normalized_features"
featuresCol="features"
partitionreal = "mappartition"
naivepartioncolname="naivepartition"


partitionnum = 5

def cost2(dict_of_df,k,querynum,partitionnum):
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
    my=np.sort(my)
    #print(my)
    c=np.sum(my,axis=0)/(querynum*k)
    return c



def calnaive(partitionnum):
    partitionnumreal=partitionnum
    l = len(traindata)
    df=pd.DataFrame(columns=['id','features',partitioncolname])#(np.arange(l),columns=['id'])
    df['id']=np.arange(l)
    df['features']=traindata.tolist()
    kmeans2 = km(n_clusters=partitionnumreal, random_state=0).fit(traindata[0:int(datalen*kmeanstrainrate)])
    res = kmeans2.predict(traindata).reshape(len(traindata),1).tolist()
    res=list(_flatten(res))
    df[naivepartioncolname] = res
    dict_of_df2 = {k: v for k, v in df.groupby(naivepartioncolname)}
    res1=[]
    ratio1=[]
    for i in range(partitionnumreal):
        l=len(dict_of_df2[i])
        res1.append(l)
    res1.sort()
    for i in range(partitionnumreal):
        ratio1.append(res1[i]/datalen)
    print("cost1 max(ratio),min(ratio),1/partitionnum",max(ratio1),min(ratio1),1/partitionnum,max(ratio1)-1/partitionnum)
    print("partitionnum",res1)
    acost2=cost2(dict_of_df2,100,querynum,partitionnum)
    print(acost2)
    print("\n\n\n\n")
    return max(ratio1)-1/partitionnum,acost2

def calcost(partitionnum,rate):
    partitionnumreal=partitionnum
    partitionnummap=int(partitionnum*rate)
    df,centroids1,centroids2=kmeansPandasDfV2(traindata,k1=partitionnumreal,k2=partitionnummap,\
            traindatanum=int(datalen*kmeanstrainrate),partitioncolname=partitioncolname)
    allpartitionrank=getallpartitionrank(centroids1,centroids2,partitionnumreal,partitionnummap)
    eachpartitonnum=geteachpartitonnum(df)
    repartitionres,repartitionnum=repartition(allpartitionrank,eachpartitonnum,partitionnumreal,partitionnummap,df.shape[0])
    sampledf_pandas=getsampledata(df,samplerate=0.05)
    partitionmap = getrepartitionmap(repartitionres,partitionnumreal,partitionnummap)
    #print("partitionmap",partitionmap)

    #df=pd.DataFrame(columns=['id','features','partition'])#(np.arange(l),columns=['id'])
    # df的 partition转化成新的partition
    def mapx(x):
        x = partitionmap[x]
        return x
    df[partitionreal]=df[partitioncolname].apply(lambda x:mapx(x))

    dict_of_df = {k: v for k, v in df.groupby(partitionreal)}
    res=[]
    ratio=[]
    for i in range(partitionnumreal):
        l=len(dict_of_df[i])
        res.append(l)

    print(res)

    res.sort()
    for i in range(partitionnumreal):
        ratio.append(res[i]/datalen)
    print("partitionnum",res)
    print("ratio",ratio)

    acost2=cost2(dict_of_df,100,querynum,partitionnum)

    print("rate",rate)
    print("cost1 max(ratio),min(ratio),1/partitionnum",max(ratio),min(ratio),1/partitionnum,max(ratio)-1/partitionnum)
    print("cost2",acost2)
    print("\n\n\n\n")
    print("rate",8)
    return max(ratio)-1/partitionnum,acost2


c1list=[]
c2list=[]
p=7
c1,c2=calnaive(p)
c1list.append(c1)
c2list.append(c2)

for i in range(2,11):
    c1,c2=calcost(p,i)
    c1list.append(c1)
    c2list.append(c2)
c3=[]
for i in c2list:
    c3.append(i[p-1])
print(c1list,"\n",c3)
