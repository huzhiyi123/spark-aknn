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
sys.path.append("/home/xxsh/xuyaoheng/newaknn")
sys.path.append("/home/xxsh/xuyaoheng/newaknn/test")

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
datapath="/home/xxsh/xuyaoheng/docker-spark/data/sift/"
traindatapath=datapath+"siftsmall_base.fvecs"
querydatapath=datapath+"siftsmall_query.fvecs"
querygroundtruthpath=datapath+"siftsmall_groundtruth.ivecs"

datapath="/home/xxsh/xuyaoheng/docker-spark/data/sift/"


"""
traindatapath=datapath+"siftsmall_base.fvecs"
querydatapath=datapath+"siftsmall_query.fvecs"
querygroundtruthpath=datapath+"siftsmall_groundtruth.ivecs"
"""

traindatapath=datapath+"sift_base.fvecs"
querydatapath=datapath+"sift_query.fvecs"
querygroundtruthpath=datapath+"sift_groundtruth.ivecs"

gistpath=datapath+"gist-960-euclidean.hdf5"

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



def ktest(): #.set('spark.jars.packages', 'com.github.jelmerk:hnswlib-spark_2.3.0_2.11:0.0.50-SNAPSHOT')
    traindata = 0
    numpyquerydata = 0
    groundtruth = 0
    partitioncolname="partition"
    centroids1name="centroids1"
    centroids2name="centroids2"
    if usesift == True:
        traindata,numpyquerydata,groundtruth = getsiftdata()
        partiname="sift"+partitioncolname
        centroids1name="sift"+centroids1name
        centroids2name="sift"+centroids2name
    else:
        traindata,numpyquerydata,groundtruth = gethdf5data()
        partiname="gist"+partitioncolname
        centroids1name="gist"+centroids1name
        centroids2name="gist"+centroids2name

    datalen=len(traindata)
    partitionnum = 8
    partitionnumreal=partitionnum
    partitionnummap=int(partitionnum*10)
    partitioncolname='partition'
    df,centroids1,centroids2=kmeansPandasDfV2(traindata,k1=partitionnumreal,k2=partitionnummap,\
            traindatanum=int(datalen*kmeanstrainrate),partitioncolname=partitioncolname)

    df[partitioncolname].to_csv(partiname+".csv", index=False)
    centroids1df=pd.DataFrame(centroids1)
    centroids1df.to_csv(centroids1name+".csv", index=False)
    centroids2df=pd.DataFrame(centroids2)
    centroids2df.to_csv(centroids2name+".csv", index=False)

usesift = False
ktest()