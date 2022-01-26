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

    c=cost2(dict_of_df2,100,querynum,partitionnum)
    print(c)
    print("\n\n\n\n")

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
    print(res)
    print("ratio",ratio)



    

    acost2=cost2(dict_of_df,100,querynum,partitionnum)

    print("rate",rate)
    print("cost1 max(ratio),min(ratio),1/partitionnum",max(ratio),min(ratio),1/partitionnum,max(ratio)-1/partitionnum)
    print("cost2",acost2)
    print("\n\n\n\n")

p=10
calnaive(p)
rate = [9]#[2,4,5,10]
for i in rate:
    calcost(p,i)
"""
nohup: ignoring input
cost1 max(ratio),min(ratio),1/partitionnum 0.1923 0.0565 0.1 0.0923
[0.000e+00 0.000e+00 0.000e+00 0.000e+00 1.000e-04 2.500e-03 2.130e-02
 6.610e-02 1.821e-01 7.279e-01]


0.0923  7.279e-01]


[850, 1247, 866, 885, 911, 679, 1414, 702, 1275, 1171]
[679, 702, 850, 866, 885, 911, 1171, 1247, 1275, 1414]
ratio [0.0679, 0.0702, 0.085, 0.0866, 0.0885, 0.0911, 0.1171, 0.1247, 0.1275, 0.1414]
rate 2
cost1 max(ratio),min(ratio),1/partitionnum 0.1414 0.0679 0.1 0.04139999999999999
cost2 [0.     0.     0.     0.     0.0025 0.0134 0.0316 0.0911 0.1988 0.6626]




[1007, 1216, 1089, 928, 815, 1142, 1076, 1015, 1063, 649]
[649, 815, 928, 1007, 1015, 1063, 1076, 1089, 1142, 1216]
ratio [0.0649, 0.0815, 0.0928, 0.1007, 0.1015, 0.1063, 0.1076, 0.1089, 0.1142, 0.1216]
rate 4
cost1 max(ratio),min(ratio),1/partitionnum 0.1216 0.0649 0.1 0.021599999999999994
cost2 [0.000e+00 0.000e+00 4.000e-04 1.700e-03 3.200e-03 1.470e-02 4.450e-02
 1.050e-01 2.246e-01 6.059e-01]




[749, 1176, 1147, 762, 1039, 1007, 1014, 1081, 1240, 785]
[749, 762, 785, 1007, 1014, 1039, 1081, 1147, 1176, 1240]
ratio [0.0749, 0.0762, 0.0785, 0.1007, 0.1014, 0.1039, 0.1081, 0.1147, 0.1176, 0.124]
rate 5
cost1 max(ratio),min(ratio),1/partitionnum 0.124 0.0749 0.1 0.023999999999999994
cost2 [0.000e+00 0.000e+00 2.000e-04 1.400e-03 4.200e-03 1.400e-02 4.550e-02
 1.105e-01 2.320e-01 5.922e-01]




[950, 1079, 945, 896, 1112, 1206, 1095, 796, 1178, 743]
[743, 796, 896, 945, 950, 1079, 1095, 1112, 1178, 1206]
ratio [0.0743, 0.0796, 0.0896, 0.0945, 0.095, 0.1079, 0.1095, 0.1112, 0.1178, 0.1206]
rate 10
cost1 max(ratio),min(ratio),1/partitionnum 0.1206 0.0743 0.1 0.020599999999999993
cost2 [0.000e+00 2.000e-04 8.000e-04 2.100e-03 7.000e-03 2.270e-02 5.570e-02
 1.133e-01 2.098e-01 5.884e-01]



[3,6,8]#[2,4,5,10]
[2,3,4,5,6,8]

0.0923  0.7279
0.04139999999999999  0.6626
0.023099999999999996 0.6022
0.021599999999999994 0.6059
0.023999999999999994 0.5922
0.010999999999999996 0.5659
0.026599999999999985 0.6079
0.018299999999999997 0.5698
0.015099999999999988 0.5791
0.020599999999999993 0.5884





"""