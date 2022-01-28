from re import S
import sys
import pandas as pd
from dis import dis
from re import S
import sys
import pandas as pd

from tempfile import gettempdir
import pandas as pd
import numpy as np
import pickle
from sklearn.cluster import KMeans as km
import sys
aknnfold="/aknn/"
sys.path.append(aknnfold)
sys.path.append(aknnfold+"test")

from sklearn.metrics import euclidean_distances
from tempfile import gettempdir
import pandas as pd
import numpy as np
from tkinter import _flatten
import pickle
from sklearn.cluster import KMeans as km
import sys
import h5py
import numpy as np
from utils import *
from datasets import *
from kmeans_repartition_utils import *
from utils import *
from datasets import *
from params import *
datapath="/sift/"
traindatapath=datapath+"siftsmall_base.fvecs"
querydatapath=datapath+"siftsmall_query.fvecs"
querygroundtruthpath=datapath+"siftsmall_groundtruth.ivecs"

#datapath="/home/xxsh/xuyaoheng/docker-spark/data/sift/"

#"/sift/"
"""
traindatapath=datapath+"siftsmall_base.fvecs"
querydatapath=datapath+"siftsmall_query.fvecs"
querygroundtruthpath=datapath+"siftsmall_groundtruth.ivecs"
"""
datapath="/data/"
traindatapath=datapath+"sift_base.fvecs"
querydatapath=datapath+"sift_query.fvecs"
querygroundtruthpath=datapath+"sift_groundtruth.ivecs"

gistpath=datapath+"mnist.hdf5" #"gist-960-euclidean.hdf5"


def gethdf5data():
    f = h5py.File(gistpath,'r+')
    keys=['distances', 'neighbors', 'test', 'train']
    print(type(f))
    print(f.keys())
    numpyquerydata =(f['test'][:])
    groundtruth = (f['neighbors'][:])
    traindata = (f['train'][:])
    #numpyquerydata =(f[2][:])
    #groundtruth = (f[1][:])
    #traindata = (f[3][:])
    return traindata,numpyquerydata,groundtruth

def getsiftdata():
    traindata = fvecs_read(traindatapath)   #.reshape(-1,128)  #[0,base*num:-1]
    numpyquerydata = fvecs_read(querydatapath)
    groundtruth = ivecs_read(querygroundtruthpath)
    return traindata,numpyquerydata,groundtruth



def ktest(partitionnumcur): #.set('spark.jars.packages', 'com.github.jelmerk:hnswlib-spark_2.3.0_2.11:0.0.50-SNAPSHOT')
    traindata = 0
    numpyquerydata = 0
    groundtruth = 0
    partitioncolname="partition" +"-" + str(partitionnumcur)
    centroids1name="centroids1" +"-" +str(partitionnumcur) 
    centroids2name="centroids2" +"-"+ str(partitionnumcur)
    if usesift == True:
        traindata,numpyquerydata,groundtruth = getsiftdata()
        partiname="sift"+partitioncolname
        centroids1name="sift"+centroids1name
        centroids2name="sift"+centroids2name
    else:
        traindata,numpyquerydata,groundtruth = gethdf5data()
        partiname="mnist"+partitioncolname
        centroids1name="mnist"+centroids1name
        centroids2name="mnist"+centroids2name

    datalen=len(traindata)
    partitionnum = partitionnumcur
    partitionnumreal=partitionnum
    partitionnummap=int(partitionnum*5)
    partitioncolname='partition'
    T1=time.time()
    df,centroids1,centroids2=kmeansPandasDfV2(traindata,k1=partitionnumreal,k2=partitionnummap,\
            traindatanum=int(datalen*kmeanstrainrate),partitioncolname=partitioncolname)
    T2=time.time()
    print("print((T2-T1)*1000)",(T2-T1)*1000,"partitionnumcur",partitionnumcur)
    df[partitioncolname].to_csv(partiname+".csv", index=False,header=False)
    centroids1df=pd.DataFrame(centroids1)
    centroids1df.to_csv(centroids1name+".csv", index=False,header=False)
    centroids2df=pd.DataFrame(centroids2)
    centroids2df.to_csv(centroids2name+".csv", index=False,header=False)

usesift = True
kmeanstrainrate=0.05
partitionnumlist = [6,10] #[4,8,12,16]
for i in partitionnumlist:
    ktest(i)