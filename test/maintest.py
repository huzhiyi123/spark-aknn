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
aknnfold="/my/"
sys.path.append(aknnfold+"spark-aknn")
from utils import *
from datasets import *
import time 
from time import sleep
findspark.init() 

# map at KnnAlgorithm.scala:507) finished in
"""
maxelement = 100000000
k=10
partitionnum=3
ef = int(1.2*k)
sc = 1
m = int(20)
distanceFunction='euclidean'
openSparkUI=True
"""
def testk(): #.set('spark.jars.packages', 'com.github.jelmerk:hnswlib-spark_2.3.0_2.11:0.0.50-SNAPSHOT')
    APP_NAME = "mytest" #setMaster("local[2]").
    conf = (SparkConf().setAppName(APP_NAME)).setSparkHome("/home/yaoheng/sparkhub/spark-2.3.0-bin-hadoop2.7")
    sc = SparkContext(conf=conf)
    sc.setCheckpointDir(gettempdir())
    sql_context = SQLContext(sc)
    partitioncolname="partitionCol"
    queryPartitionsCol='querypartitions'
    nfcolname="normalized_features"
    # 分区并且 训练分布式hnsw 这里没有归一化
    df = kmeansPandasDf(traindatapath,k=8,traindatanum=2000)
    # id features partition
    curschema = StructType([StructField("id", IntegerType()),StructField("features",ArrayType(DoubleType())),StructField(partitioncolname, IntegerType() )])
    traindata_df = sql_context.createDataFrame(df,curschema)
    traindata_df.printSchema()
    sc.stop()
    print("hello world SparkHnsw\n")



    return conf

def SparkHnsw(): #.set('spark.jars.packages', 'com.github.jelmerk:hnswlib-spark_2.3.0_2.11:0.0.50-SNAPSHOT')
    print("SparkHnsw():")
    APP_NAME = "mytest" #setMaster("local[2]").
    conf = (SparkConf().setAppName(APP_NAME))#.setSparkHome("/home/yaoheng/sparkhub/spark-2.3.0-bin-hadoop2.7")
    conf = setconf(conf)
    sc = SparkContext(conf=conf)
    sc.setCheckpointDir(gettempdir())
    sql_context = SQLContext(sc)
    partitioncolname="partitionCol"
    queryPartitionsCol='querypartitions'
    nfcolname="normalized_features"
    featuresCol="features"
    # 分区并且 训练分布式hnsw 这里没有归一化
    traindata = fvecs_read(traindatapath)[0:300]   #.reshape(-1,128)  #[0,base*num:-1]
    #print(type(traindata),traindata.shape)
    # 2 4 6 8
    T1 = time.time()
    datalen=len(traindata)
    df = kmeansPandasDf(traindata,k=partitionnum,traindatanum=int(datalen*0.1))
    T2 = time.time()
    kmeanstime=(T2-T1)*1000
    #print("kmeanstimepartitiontime",kmeanstime)
    # id features partition
    curschema = StructType([StructField("id", IntegerType()),StructField("features",ArrayType(DoubleType())),StructField(partitioncolname, IntegerType() )])
    words_df = sql_context.createDataFrame(df,curschema)
    #words_df.show()
    #,queryPartitionsCol=queryPartitionsCol
    hnsw = HnswSimilarity(identifierCol='id',queryIdentifierCol='id',featuresCol=featuresCol, 
                        distanceFunction=distanceFunction, m=m, ef=ef, k=k, efConstruction=ef, numPartitions=partitionnum, 
                        excludeSelf=True, predictionCol='approximate', outputFormat='minimal')
    hnsw.setPartitionCol(partitioncolname)
    
    T3 = time.time()
    #print("word_df.count()",words_df.count())
    model=hnsw.fit(words_df)
    T4 = time.time()
    localindexconstructtime=(T4-T3)*1000
    #print("localindexconstruct=(T4-T3)*1000",localindexconstructtime)

    sampledf = resample_in_partition(words_df,kmeanstrainrate,partitioncolname)
    sampledf_pandas = sampledf.toPandas()
    #def hnsw_global_index_pddf(pddf,max_elements,dim,nf_col="normalized_features",partitionid_col="partition_id"):
    hnsw_global_model = hnsw_global_index_pddf(sampledf_pandas,1000000,128,featurecol=featuresCol,partitionid_col=partitioncolname)
    T5 = time.time()
    globalindeconstructtime=(T5-T4)*1000
    #print("globalindeconstructtime=(T5-T4)*1000",globalindeconstructtime)

    # 读取查询向量 并且全局索引查询预测的分区
    numpyquerydata = fvecs_read(querydatapath)
    # id features(arrary) partioncol(int)

    # 这里处理
    queryvec,globalserchtime = processQueryVec(hnsw_global_model,numpyquerydata,sampledf_pandas,queryPartitionsCol,\
                                partitionCol=partitioncolname,partitionnum=partitionnum,\
                                topkPartitionNum=topkPartitionNum,knnQueryNum=30)    


    curschema = StructType([ StructField("id", IntegerType() ),StructField(featuresCol,ArrayType(DoubleType())),StructField(queryPartitionsCol,ArrayType(IntegerType()))])
    query_df = sql_context.createDataFrame(queryvec,curschema)
    #query_df.printSchema()
    # todo with index
    T6 = time.time()
    print("start query")
    result=model.transform(query_df).orderBy("id")
    #print("result.count()",result.count())
    result.count()
    print("end query")
    T7 = time.time()
    
    localsearchtime=(T7-T6)*1000
    #print("localsearchtime",localsearchtime)
    predict = processSparkDfResult(result)
    #print("result.count()",result.count())
    

    groundtruth = ivecs_read(querygroundtruthpath)
    #print("groundtruth[0:5]:",groundtruth[:,0:k])
    if(len(predict) != 0):
        recall1 = evaluatePredict(predict,groundtruth,k)
        print("recall:",recall1)
    #if not openSparkUI:
    #sleep(150)
    sc.stop()
    #input("ffew")
    print("hello world SparkHnsw\n")
    totalsearchtime=localsearchtime+globalserchtime
    print("searchtimeUsed: ",totalsearchtime,"globalindextime",globalserchtime,"localsearchtime",localsearchtime)
    totalconstructtime=kmeanstime+localindexconstructtime+globalindeconstructtime
    print("totalconstructtime",totalconstructtime,\
        "kmeanstime",kmeanstime,"localindexconstructtime",localindexconstructtime,\
        "globalindeconstructtime",globalindeconstructtime)
    #return recall1

def readDataSparkDfquery(sql_context,traindatapath,num):
        # 读取查询向量 并且全局索引查询预测的分区
    qd = fvecs_read(traindatapath)[0:num]
    # id features(arrary) partioncol(int)
    vec = pd.DataFrame(np.arange(qd.shape[0]),columns=["id"])
    vec['normalized_features'] = qd.tolist()
    curschema = StructType([ StructField("id", IntegerType() ),StructField("normalized_features",ArrayType(DoubleType()))])
    df = sql_context.createDataFrame(vec,curschema)
    return df

def readDataSparkDfquery2(sql_context,traindatapath):
        # 读取查询向量 并且全局索引查询预测的分区
    qd = fvecs_read(traindatapath)
    #qd = np.concatenate((qd,qd))
    qd = np.concatenate((qd,qd))
    # id features(arrary) partioncol(int)
    vec = pd.DataFrame(np.arange(qd.shape[0]),columns=["id"])
    vec['normalized_features'] = qd.tolist()
    curschema = StructType([ StructField("id", IntegerType() ),StructField("normalized_features",ArrayType(DoubleType()))])
    df = sql_context.createDataFrame(vec,curschema)
    return df



def bruteForce(): #.set('spark.jars.packages', 'com.github.jelmerk:hnswlib-spark_2.3.0_2.11:0.0.50-SNAPSHOT')
    APP_NAME = "mytest" #setMaster("local[2]").
    conf = (SparkConf().setAppName(APP_NAME)) #.setSparkHome("/home/yaoheng/sparkhub/spark-2.3.0-bin-hadoop2.7")
    sc = SparkContext(conf=conf)
    sc.setCheckpointDir(gettempdir())
    sql_context = SQLContext(sc)
    partitioncolname="partitionCol"
    queryPartitionsCol='querypartitions'
    nfcolname="normalized_features"
    # 分区并且 训练分布式hnsw 这里没有归一化
    traindf = readDataSparkDf(sql_context,traindatapath)
    # 总共1w条
    querydf = readDataSparkDfquery(sql_context,querydatapath,500)
    #querydf = readDataSparkDfquery(sql_context,querydatapath,1000) # 8分区 vs 4分区
    bruteforce = BruteForceSimilarity(identifierCol='id', queryIdentifierCol='id', featuresCol='normalized_features',
                                    distanceFunction=distanceFunction, numPartitions=partitionnum, excludeSelf=False,
                                    predictionCol='approximate', outputFormat='minimal',k=k)
    T3 = time.time()
    model = bruteforce.fit(traindf)
    T1 = time.time()
    predict = model.transform(querydf).orderBy("id")
    print("predict.count()",predict.count())
    T2 = time.time()
    timeUsed = (T2-T1)*1000
    print("timeUsed: ",timeUsed,"fit time",(T1-T3)*1000)
    sc.stop()
    print("done bruteForce\n")
    return 0


def testmain_naiveSparkHnsw(): #.set('spark.jars.packages', 'com.github.jelmerk:hnswlib-spark_2.3.0_2.11:0.0.50-SNAPSHOT')
    APP_NAME = "mytest" #setMaster("local[2]").
    conf = (SparkConf().setAppName(APP_NAME)).setSparkHome("/home/yaoheng/sparkhub/spark-2.3.0-bin-hadoop2.7")
    sc = SparkContext(conf=conf)
    sc.setCheckpointDir(gettempdir())
    sql_context = SQLContext(sc)
    # 分区并且 训练分布式hnsw 这里没有归一化
    words_df = readDataSparkDf(sql_context,traindatapath)
    hnsw = HnswSimilarity(identifierCol='id', queryIdentifierCol='id',featuresCol='normalized_features',
                         distanceFunction=distanceFunction,m=m, ef=ef, k=k,efConstruction=ef,
                         numPartitions=partitionnum,  predictionCol='approximate',excludeSelf=True)
    model=hnsw.fit(words_df)

    # 读取查询向量 并且全局索引查询预测的分区
    query_df = readDataSparkDf(sql_context,querydatapath)
    #print("query_df.printSchema()")
    query_df.printSchema()
    query_df.show()
    # todo with index
    T1 = time.time()
    result=model.transform(query_df)
    T2 = time.time()
    timeUsed = (T2-T1)*1000
    print("timeUsed: ",timeUsed)
    #print("result.printSchema()")
    #result.printSchema()
    result = result.orderBy("id")
    result.count()
    predict = processSparkDfResult(result)
    sc.stop()
"""
if __name__ == "__main__":
    efConstructionlist = [20,40,80,200]
    initparams()
    print("for i in efConstructionlist:")
    for i in efConstructionlist:
        initparams()
        efConstruction = i
        print("efConstruction cmp",efConstruction)
        SparkHnsw()
    print("end efConstructionlist\n",efConstructionlist)

"""
"""
if __name__ == "__main__":
    
    klist = [5,10,20,30,40,50]
    efmullist = [1.5,3,4,5,8,10,12,15]
    mlist = [10,25,30,40,50]
    efConstructionlist = [20,50,100,200,300,400]

    initparams()
    print("for ki in klist:")
    for i in klist:
        print("k cmp",i)
        k=i
        SparkHnsw()
    print("end klist\n",klist)

    initparams()
    print("for i in efmul:")
    for i in efmullist:
        ef = int(i*k)
        print("ef cmp",ef)
        SparkHnsw()
    print("end efmullist\n",efmullist)


    initparams()
    print("for i in mlist:")
    for i in mlist:
        print("m cmp",i)
        m=i
        SparkHnsw()
    print("end mlist\n",mlist)


    initparams()
    print("for i in efConstructionlist:")
    for i in efConstructionlist:
        efConstruction = i
        print("efConstruction cmp",efConstruction)
        SparkHnsw()
    print("end efConstructionlist\n",efConstructionlist)
# k 的对比 ef
"""