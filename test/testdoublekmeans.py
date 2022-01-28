# coding=utf-8
from pyspark_hnsw.linalg import Normalizer
from pyspark_hnsw.knn import HnswSimilarity
from pyspark.ml.linalg import Vectors
from pyspark.sql import SQLContext
# $example on$
from pyspark.ml import Pipeline
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.feature import HashingTF, Tokenizer
# $example off$
from pyspark.sql import SparkSession
from pyspark import SparkContext, SparkConf
from pyspark.ml.feature import VectorAssembler
from pyspark_hnsw.conversion import VectorConverter
import findspark
from tempfile import gettempdir
import pandas as pd
from pyspark_hnsw.evaluation import KnnSimilarityEvaluator
from pyspark.ml.linalg import Vectors
from pyspark.sql import SQLContext,DataFrame
from pyspark.sql.types import *
from pyspark_hnsw.knn import BruteForceSimilarity
from pyspark.sql.functions import array, col
import numpy as np
from params import *
from pyspark.mllib.clustering import KMeans,KMeansModel
import pyspark.sql.functions as F
import hnswlib
from sklearn.preprocessing import normalize
import sys
aknnfold="/aknn/"
sys.path.append(aknnfold)
sys.path.append(aknnfold+"test")
from utils import *
from datasets import *
from kmeans_repartition_utils import *
from maintest import *

import time 
from time import sleep

import h5py
import numpy as np


findspark.init() 

def initparams():
    global maxelement,k,partitionnum,topkPartitionNum,ef,m,distanceFunction,kmeanstrainrate,efConstruction,usesift
    maxelement = 100000000
    k=10
    partitionnum=8
    topkPartitionNum=3
    sc = 1
    m = int(50)
    distanceFunction='cosine'
    kmeanstrainrate=0.05
    efConstruction=100
    ef = efConstruction
    usesift=True


traindata = 0
numpyquerydata = 0
groundtruth = 0

gistpath="/data/mnist.hdf5"


rate = 5


def gethdf5data(gistpath):
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

def kmeansPandasDfV3(data,partitioncsvpath,centroids1path,centroids2path,partitioncolname="partition"):
    l = len(data)
    df=pd.DataFrame(columns=['id','features',partitioncolname])#(np.arange(l),columns=['id'])
    df['id']=np.arange(l)
    df['features']=data.tolist()
    tmp=pd.read_csv(partitioncsvpath,header=None,index_col=None).values
    if(len(tmp) < l):
        lenghth = l-len(tmp)
        zeros = np.zeros(lenghth).reshape(lenghth,1).astype(np.int)
        tmp = np.concatenate((tmp,zeros),axis=0)
    print("type(tmp),tmp[0:100],tmp.shape",type(tmp),tmp[0:5],tmp.shape,df.shape) 
    df[partitioncolname]=tmp.astype(np.int)
    centroids1 = pd.read_csv(centroids1path,header=None,index_col=None).values
    centroids2 = pd.read_csv(centroids2path,header=None,index_col=None).values
    return df,centroids1,centroids2


def testdoublekmeansHnsw(): #.set('spark.jars.packages', 'com.github.jelmerk:hnswlib-spark_2.3.0_2.11:0.0.50-SNAPSHOT')
    APP_NAME = "mytest" #setMaster("local[2]").
    conf = (SparkConf().setAppName(APP_NAME))#.setSparkHome("/home/yaoheng/sparkhub/spark-2.3.0-bin-hadoop2.7")
    sc = SparkContext(conf=conf)
    sc.setCheckpointDir(gettempdir())
    sql_context = SQLContext(sc)
    partitioncolname="partition"
    queryPartitionsCol='querypartitions'
    nfcolname="normalized_features"
    featuresCol="features"
    partitionreal = "mappartition"
    # 分区并且 训练分布式hnsw 这里没有归一化
    """
    traindata = fvecs_read(traindatapath)   #.reshape(-1,128)  #[0,base*num:-1]
    numpyquerydata = fvecs_read(querydatapath)
    groundtruth = ivecs_read(querygroundtruthpath)
    #print(type(traindata),traindata.shape)
    """
    traindata = 0
    numpyquerydata = 0
    groundtruth = 0
    dimensionality = 0
    if usesift == True:
        traindata,numpyquerydata,groundtruth = getsiftdata()
        dimensionality = 128
    else:
        traindata,numpyquerydata,groundtruth = gethdf5data(gistpath)
        dimensionality = 784
    print("(traindata.shape)",traindata.shape)
    # 2 4 6 8

    datalen=len(traindata)

    partitionnumreal=partitionnum
    partitionnummap=int(partitionnum*rate)
    

    T1 = time.time()
    df,centroids1,centroids2=kmeansPandasDfV2(traindata,k1=partitionnumreal,k2=partitionnummap,\
        traindatanum=int(datalen*kmeanstrainrate),partitioncolname=partitioncolname)
    # df 的 partition col映射到真是的cols
    # df=pd.DataFrame(columns=['id','features','partition'])#(np.arange(l),columns=['id'])
    T2 = time.time()
    kmeanspartitiontime=(T2-T1)*1000
    #print("kmeanspartitiontime",kmeanspartitiontime)

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
    #print(df[0:20])


    # id features partition
    curschema = StructType([StructField("id", IntegerType()),
    StructField("features",ArrayType(DoubleType())),
    StructField(partitioncolname, IntegerType() ),
    StructField(partitionreal, IntegerType() )
    ])
    words_df = sql_context.createDataFrame(df,curschema)
    #words_df.show()
    #,
    hnsw = HnswSimilarity(identifierCol='id',queryIdentifierCol='id',featuresCol=featuresCol,queryPartitionsCol=queryPartitionsCol,
                        distanceFunction=distanceFunction, m=m, ef=ef, k=k, efConstruction=efConstruction, numPartitions=partitionnum, 
                        excludeSelf=True, predictionCol='approximate', outputFormat='minimal')
    hnsw.setPartitionCol(partitionreal)
    T3 = time.time()
    #print("word_df.count()",words_df.count())
    model=hnsw.fit(words_df)
    T4 = time.time()
    localindexconstructtime=(T4-T3)*1000
    #print("localindexconstructtime",localindexconstructtime)
    #print("model=hnsw.fit(words_df) done")
    T5 = time.time()
    #def hnsw_global_index_pddf(pddf,max_elements,dim,nf_col="normalized_features",partitionid_col="partition_id"):
    hnsw_global_model=hnsw_global_index_wrapper(sampledf_pandas,1000000,dimensionality,featurecol='features')
    T6 = time.time()
    globalindexconstructtime=(T6-T5)*1000
    #print("globalindexconstructtime",globalindexconstructtime)

    # 读取查询向量 并且全局索引查询预测的分区
    
    # id features(arrary) partioncol(int)

    # 这里处理
    # sampledf_pandas: 'id','features','partition'
    knnQueryNum=30
    queryvec,globalsearchtime = processQueryVecv2(hnsw_global_model,numpyquerydata,\
    sampledf_pandas,queryPartitionsCol,partitioncolname,partitionmap,\
    partitionnum=partitionnum,topkPartitionNum=topkPartitionNum,knnQueryNum=knnQueryNum)
    #print("globalsearchtime",globalsearchtime)


    curschema = StructType([ StructField("id", IntegerType() ),\
        StructField(featuresCol,ArrayType(DoubleType())),
        StructField(queryPartitionsCol,ArrayType(IntegerType()))])
    query_df = sql_context.createDataFrame(queryvec,curschema)
    #query_df.printSchema()
    # todo with index
    T7 = time.time()
    print("start query")
    result=model.transform(query_df).orderBy("id")
    print("result.count()",result.count())
    print("end query")
    T8 = time.time()
    localsearchtime=(T8-T7)*1000
    #print("localsearchtime",localsearchtime)

    predict = processSparkDfResult(result)
    #print("predict = processSparkDfResult(result)",predict)

    
    #print("groundtruth[0:5]:",groundtruth[:,0:k])
    if(len(predict) != 0):
        recall1 = evaluatePredict(predict,groundtruth,k)
        print("recall:",recall1)
    #if not openSparkUI:
    #sleep(150)
    sc.stop()
    #input("ffew")
    totalsearchtime = localsearchtime + globalsearchtime
    totalconstructtime=kmeanspartitiontime+localindexconstructtime+globalindexconstructtime
    
    print("totalconstructtime",totalconstructtime,\
        "kmeanspartitiontime",kmeanspartitiontime,"localindexconstructtime",\
            localindexconstructtime,"globalindexconstructtime",globalindexconstructtime)

    print("totalsearchtime",totalsearchtime,"localsearchtime",localsearchtime,"globalsearchtime",globalsearchtime)

    #return recall1

def readDataSparkDfv3(sql_context,qd):
    vec = pd.DataFrame(np.arange(qd.shape[0]),columns=["id"])
    vec['normalized_features'] = qd.tolist()
    curschema = StructType([ StructField("id", IntegerType() ),StructField("normalized_features",ArrayType(DoubleType()))])
    df = sql_context.createDataFrame(vec,curschema)
    return df

def bruteForce(): #.set('spark.jars.packages', 'com.github.jelmerk:hnswlib-spark_2.3.0_2.11:0.0.50-SNAPSHOT')
    APP_NAME = "mytest" #setMaster("local[2]").
    conf = (SparkConf().setAppName(APP_NAME))
    sc = SparkContext(conf=conf)
    sc.setCheckpointDir(gettempdir())
    sql_context = SQLContext(sc)
    partitioncolname="partitionCol"
    queryPartitionsCol='querypartitions'
    nfcolname="normalized_features"
    # 分区并且 训练分布式hnsw 这里没有归一化
    traindata = 0
    numpyquerydata = 0
    groundtruth = 0
    if usesift == True:
        traindata,numpyquerydata,groundtruth = getsiftdata()
    else:
        traindata,numpyquerydata,groundtruth = gethdf5data(gistpath)
    
    traindf = readDataSparkDfv3(sql_context,traindata)
    querydf = readDataSparkDfv3(sql_context,numpyquerydata)

    bruteforce = BruteForceSimilarity(identifierCol='id', queryIdentifierCol='id', featuresCol='normalized_features',
                                    distanceFunction=distanceFunction, numPartitions=partitionnum, excludeSelf=False,
                                    predictionCol='approximate', outputFormat='minimal',k=k)
    T3 = time.time()
    model = bruteforce.fit(traindf)
    T1 = time.time()
    predict = model.transform(querydf).orderBy("id")
    print("predict.count()",predict.count())
    predict.printSchema()
    T2 = time.time()
    timeUsed = (T2-T1)*1000
    print("timeUsed: ",timeUsed,"fit time",(T1-T3)*1000)
    """
    groundtruth = ivecs_read(querygroundtruthpath)[:,:k]
    predict = processSparkDfResult(predict)
    recall = evaluatePredict(predict,groundtruth,k)
    print("recall",recall)
    print(predict)
    print(groundtruth)
    """
    sc.stop()
    print("hello world bruteForce\n")
    return 0



def testdoublekmeansHnswV2(): #.set('spark.jars.packages', 'com.github.jelmerk:hnswlib-spark_2.3.0_2.11:0.0.50-SNAPSHOT')
    APP_NAME = "mytest" #setMaster("local[2]").
    conf = (SparkConf().setAppName(APP_NAME))#.setSparkHome("/home/yaoheng/sparkhub/spark-2.3.0-bin-hadoop2.7")
    sc = SparkContext(conf=conf)
    sc.setCheckpointDir(gettempdir())
    sql_context = SQLContext(sc)
    partitioncolname="partition"
    queryPartitionsCol='querypartitions'
    nfcolname="normalized_features"
    featuresCol="features"
    partitionreal = "mappartition"


    traindata = 0
    numpyquerydata = 0
    groundtruth = 0
    partitionpath=""
    centroids1path=""
    centroids2path=""

    kmeanspath="/aknn/kmeans/"
    gistlist=["gistpartition.csv","gistcentroids2.csv","gistcentroids1.csv"]  
    
    mnistlist=["mnistpartition.csv","mnistcentroids1.csv","mnistcentroids2.csv"]
    mnistlistv1=["mnistpartitionv1.csv","mnistcentroids1v1.csv","mnistcentroids2v1.csv"]
    siftlist =[]


    siftlist8=["siftpartition-8.csv","siftcentroids1-8.csv","siftcentroids2-8.csv"]
    siftlist16=["siftpartition-16.csv","siftcentroids1-16.csv","siftcentroids2-16.csv"]
    siftlist6=["siftpartition-6.csv","siftcentroids1-6.csv","siftcentroids2-6.csv"]

    partitionnumreal=partitionnum
    partitionnummap=int(partitionnum*rate)
    siftlist4=["siftpartition-4.csv","siftcentroids1-4.csv","siftcentroids2-4.csv"]
    if partitionnum == 4: 
        siftlist = siftlist4
    if partitionnum == 8: 
        siftlist = siftlist8
    if partitionnum == 6:
        siftlist = siftlist6


    print("partitionnumreal,partitionnummap",partitionnumreal,partitionnummap)

    dimensionality=0
    if usesift == True:
        traindata,numpyquerydata,groundtruth = getsiftdata()
        partitionpath = kmeanspath+siftlist[0]
        centroids1path = kmeanspath+siftlist[1]
        centroids2path = kmeanspath+siftlist[2]
        dimensionality=128
    else:
        traindata,numpyquerydata,groundtruth = gethdf5data(gistpath)
        partitionpath = kmeanspath+mnistlistv1[0]
        centroids1path = kmeanspath+mnistlistv1[1]
        centroids2path = kmeanspath+mnistlistv1[2]
        dimensionality=784

        partitionnumreal=partitionnum
        partitionnummap=int(partitionnum*rate)

    datalen=len(traindata)
    


    T1 = time.time()
    df,centroids1,centroids2 = kmeansPandasDfV3(traindata,partitionpath,centroids1path,centroids2path,"partition")

    print("centroids1.shape,centroids2.shape",centroids1.shape,centroids2.shape)
    # df 的 partition col映射到真是的cols
    # df=pd.DataFrame(columns=['id','features','partition'])#(np.arange(l),columns=['id'])
    T2 = time.time()
    kmeanspartitiontime=(T2-T1)*1000
    #print("kmeanspartitiontime",kmeanspartitiontime)
    print("centroids1",centroids1.shape)
    print("centroids2",centroids2.shape)
    allpartitionrank=getallpartitionrank(centroids1,centroids2,partitionnumreal,partitionnummap)
    eachpartitonnum=geteachpartitonnum(df)
    repartitionres,repartitionnum=repartition(allpartitionrank,eachpartitonnum,partitionnumreal,partitionnummap,df.shape[0])
    sampledf_pandas=getsampledata(df,samplerate=0.05)
    partitionmap = getrepartitionmap(repartitionres,partitionnumreal,partitionnummap)
    print(len(partitionmap))
    #print("partitionmap",partitionmap)
    print("len(repartitionres)",len(repartitionres))
    print("partitionnumreal,partitionnummap,len(partitionmap)",partitionnumreal,partitionnummap,len(partitionmap))



    def mapx(x):
        #print("def mapx(x):")
        #print(type(x),x)
        x = partitionmap[x]
        return x

    df[partitionreal]=df[partitioncolname].apply(lambda x:mapx(x))

    curschema = StructType([StructField("id", IntegerType()),
    StructField("features",ArrayType(DoubleType())),
    StructField(partitioncolname, IntegerType() ),
    StructField(partitionreal, IntegerType() )
    ])
    words_df = sql_context.createDataFrame(df,curschema)
    hnsw = HnswSimilarity(identifierCol='id',queryIdentifierCol='id',featuresCol=featuresCol,queryPartitionsCol=queryPartitionsCol,
                        distanceFunction=distanceFunction, m=m, ef=ef, k=k, efConstruction=efConstruction, numPartitions=partitionnum, 
                        excludeSelf=True, predictionCol='approximate', outputFormat='minimal')
    hnsw.setPartitionCol(partitionreal)
    T3 = time.time()
    #print("word_df.count()",words_df.count())
    model=hnsw.fit(words_df)
    T4 = time.time()
    localindexconstructtime=(T4-T3)*1000
    #print("localindexconstructtime",localindexconstructtime)
    #print("model=hnsw.fit(words_df) done")
    T5 = time.time()
    #def hnsw_global_index_pddf(pddf,max_elements,dim,nf_col="normalized_features",partitionid_col="partition_id"):
    hnsw_global_model=hnsw_global_index_wrapper(sampledf_pandas,1000000,dimensionality,featurecol='features')
    T6 = time.time()
    globalindexconstructtime=(T6-T5)*1000
    #print("globalindexconstructtime",globalindexconstructtime)

    # 读取查询向量 并且全局索引查询预测的分区
    
    # id features(arrary) partioncol(int)

    # 这里处理
    # sampledf_pandas: 'id','features','partition'
    knnQueryNum=30
    queryvec,globalsearchtime = processQueryVecv2(hnsw_global_model,numpyquerydata,\
    sampledf_pandas,queryPartitionsCol,partitioncolname,partitionmap,\
    partitionnum=partitionnum,topkPartitionNum=topkPartitionNum,knnQueryNum=knnQueryNum)
    #print("globalsearchtime",globalsearchtime)


    curschema = StructType([ StructField("id", IntegerType() ),\
        StructField(featuresCol,ArrayType(DoubleType())),
        StructField(queryPartitionsCol,ArrayType(IntegerType()))])
    query_df = sql_context.createDataFrame(queryvec,curschema)
    #query_df.printSchema()
    # todo with index
    T7 = time.time()
    print("start query")
    result=model.transform(query_df).orderBy("id")
    print("result.count()",result.count())
    print("end query")
    T8 = time.time()
    localsearchtime=(T8-T7)*1000
    #print("localsearchtime",localsearchtime)

    predict = processSparkDfResult(result)
    #print("predict = processSparkDfResult(result)",predict)

    
    #print("groundtruth[0:5]:",groundtruth[:,0:k])
    if(len(predict) != 0):
        recall1 = evaluatePredict(predict,groundtruth,k)
        print("recall:",recall1)
    #if not openSparkUI:
    #sleep(150)
    sc.stop()
    #input("ffew")
    totalsearchtime = localsearchtime + globalsearchtime
    totalconstructtime=kmeanspartitiontime+localindexconstructtime+globalindexconstructtime
    
    print("totalconstructtime",totalconstructtime,\
        "kmeanspartitiontime",kmeanspartitiontime,"localindexconstructtime",\
            localindexconstructtime,"globalindexconstructtime",globalindexconstructtime)
    
    print("totalsearchtime",totalsearchtime,"localsearchtime",localsearchtime,"globalsearchtime",globalsearchtime)
    print("hello world testdoublekmeansHnsw\n")
    #return recall1
 
if __name__ == "__main__":
    rate = 5
    initparams()
    print("maxelement,k,partitionnum,topkPartitionNum,ef,m,distanceFunction,kmeanstrainrate,efConstruction,usesift\n",
    maxelement,k,partitionnum,topkPartitionNum,ef,m,distanceFunction,kmeanstrainrate,efConstruction,usesift)
    #usesift = False
    efConstructionlist = [15,20,50,100,150]
    # efConstructionlist = [12,15,18,20,30,50,100,200]
    for i in efConstructionlist:
        initparams()
        efConstruction = i
        ef = efConstruction
        usesift = True
        print("topkPartitionNum cmp",i)
        testdoublekmeansHnswV2()
    print("end topkPartitionNum cmp\n")



"""
    klist = [5,10,20,30,40,50]
    for i in klist:
        initparams()
        topkPartitionNum=3
        efConstruction = 100
        ef = efConstruction
        usesift = True
        k=i
        print("klist cmp",k)
        testdoublekmeansHnswV2()
    print("end klist cmp\n",k)
"""
# TODO 
# topk partition


