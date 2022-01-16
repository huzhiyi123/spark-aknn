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
datapath="/my/siftsmall/"
traindatapath=datapath+"siftsmall_base.fvecs"
querydatapath=datapath+"siftsmall_query.fvecs"
querygroundtruthpath=datapath+"siftsmall_groundtruth.ivecs"
"""
traindatapath="/home/yaoheng/test/data/sift/sift_base.fvecs"
querydatapath="/home/yaoheng/test/data/sift/sift_query.fvecs"
querygroundtruthpath="/home/yaoheng/test/data/sift/sift_groundtruth.ivecs"
"""
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


def setconf(conf):
    conf.set("spark.executor.memory", "1g")
    conf.set("spark.driver.memory","10g")
    conf.set("spark.executor.cores","2")
    conf.set("spark.driver.maxResultSize","2g")
    conf.set("spark.dynamicAllocation.enabled","true")
    conf.set("spark.shuffle.service.enabled", "true")
    conf.set("spark.dynamicAllocation.maxExecutors","8")
    conf.set("executor.instances","8")
    conf.set("spark.task.maxFailures","1")
    conf.set("spark.default.parallelism","8")
    return conf

def SparkHnsw(): #.set('spark.jars.packages', 'com.github.jelmerk:hnswlib-spark_2.3.0_2.11:0.0.50-SNAPSHOT')
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
    traindata = fvecs_read(traindatapath)   #.reshape(-1,128)  #[0,base*num:-1]
    print(type(traindata),traindata.shape)
    # 2 4 6 8
    T10 = time.time()
    df = kmeansPandasDf(traindata,k=partitionnum,traindatanum=3000)
    T11 = time.time()
    # id features partition
    curschema = StructType([StructField("id", IntegerType()),StructField("features",ArrayType(DoubleType())),StructField(partitioncolname, IntegerType() )])
    words_df = sql_context.createDataFrame(df,curschema)

    #,queryPartitionsCol=queryPartitionsCol
    hnsw = HnswSimilarity(identifierCol='id',queryPartitionsCol=queryPartitionsCol,queryIdentifierCol='id',featuresCol=featuresCol, 
                        distanceFunction=distanceFunction, m=m, ef=ef, k=k, efConstruction=ef, numPartitions=partitionnum, 
                        excludeSelf=True, predictionCol='approximate', outputFormat='minimal')
    hnsw.setPartitionCol(partitioncolname)

    T5 = time.time()
    print("word_df.count()",words_df.count())
    model=hnsw.fit(words_df)
    
    T12 = time.time()
    sampledf = resample_in_partition(words_df,0.05,partitioncolname)
    sampledf_pandas = sampledf.toPandas()
    #def hnsw_global_index_pddf(pddf,max_elements,dim,nf_col="normalized_features",partitionid_col="partition_id"):
    hnsw_global_model = hnsw_global_index_pddf(sampledf_pandas,1000000,128,featurecol=featuresCol,partitionid_col=partitioncolname)
    T13 = time.time()
    T6 = time.time()
    # 读取查询向量 并且全局索引查询预测的分区
    numpyquerydata = fvecs_read(querydatapath)
    numpyquerydata = numpyquerydata
    # id features(arrary) partioncol(int)
    T3 = time.time()
    # 这里处理
    queryvec = processQueryVec(hnsw_global_model,numpyquerydata,sampledf_pandas,queryPartitionsCol,partitionnum=partitionnum,topkPartitionNum=topkPartitionNum,knnQueryNum=10)

    print("numpyquerydata.shape",numpyquerydata.shape,"queryvec.shape",queryvec.shape)
    T4 = time.time()
    curschema = StructType([ StructField("id", IntegerType() ),StructField(featuresCol,ArrayType(DoubleType())),StructField(queryPartitionsCol,ArrayType(IntegerType()))])
    query_df = sql_context.createDataFrame(queryvec,curschema)
    #query_df.printSchema()
    # todo with index
    T1 = time.time()
    print("start query")
    result=model.transform(query_df).orderBy("id")
    print("result.count()",result.count())
    result.show()
    print("end query")
    T2 = time.time()
    predict = processSparkDfResult(result)
    print("predict = processSparkDfResult(result)",predict)
    print("result.count()",result.count())
    result.show()
    timeUsed = (T2-T1)*1000
    print("timeUsed: ",timeUsed,"globalindextime",(T4-T3)*1000,"construct time",(T6-T5)*1000)
    
    print("kmeans-time",(T11-T10)*1000)
    print("globalindex-construct-time",(T13-T12)*1000)
    groundtruth = ivecs_read(querygroundtruthpath)
    #print("groundtruth[0:5]:",groundtruth[:,0:k])
    if(len(predict) != 0):
        recall1 = evaluatePredict(predict,groundtruth,k)
        print("recall:",recall1)
    #if not openSparkUI:
    sleep(40)
    sc.stop()
    #input("ffew")
    print("hello world SparkHnsw\n")
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
    conf = (SparkConf().setAppName(APP_NAME)).setSparkHome("/home/yaoheng/sparkhub/spark-2.3.0-bin-hadoop2.7")
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


def testmain_naiveSparkHnsw(): #.set('spark.jars.packages', 'com.github.jelmerk:hnswlib-spark_2.3.0_2.11:0.0.50-SNAPSHOT')
    APP_NAME = "mytest" #setMaster("local[2]").
    conf = (SparkConf().setAppName(APP_NAME)).setSparkHome("/home/yaoheng/sparkhub/spark-2.3.0-bin-hadoop2.7")
    sc = SparkContext(conf=conf)
    sc.setCheckpointDir(gettempdir())
    sql_context = SQLContext(sc)
    # 分区并且 训练分布式hnsw 这里没有归一化
    words_df = readDataSparkDf(sql_context,traindatapath)
    """
    print("words_df.printSchema()")
    words_df.printSchema()
    print("words_df.printSchema()")
    words_df.printSchema()
    """
    hnsw = HnswSimilarity(identifierCol='id', queryIdentifierCol='id',featuresCol='normalized_features',
                         distanceFunction=distanceFunction,m=m, ef=ef, k=k,efConstruction=ef,
                         numPartitions=partitionnum,  predictionCol='approximate',excludeSelf=True)
    model=hnsw.fit(words_df)

    # 读取查询向量 并且全局索引查询预测的分区
    query_df = readDataSparkDf(sql_context,querydatapath)
    print("query_df.printSchema()")
    query_df.printSchema()
    query_df.show()
    # todo with index
    T1 = time.time()
    result=model.transform(query_df)
    T2 = time.time()
    timeUsed = (T2-T1)*1000
    print("timeUsed: ",timeUsed)
    print("result.printSchema()")
    result.printSchema()
    result = result.orderBy("id")
    result.show()
    predict = processSparkDfResult(result)
    sc.stop()

if __name__ == "__main__":
    #testmain()
    """
    a=SparkHnsw()
    testmain_naiveSparkHnsw()
    b=bruteForce()
    """
    #recall1=SparkHnsw()
    #print(recall1)
    
    #num = [2,4,6,8]
    #for i in num:
    SparkHnsw()

    #recall=bruteForce() #SparkHnsw() #
    #print(recall)
    
    
    #print(a,b)
    #testmain_naiveSparkHnsw()
    #bruteForce()
    #text = input("Please enter a text:")
    #test()

# cat /home/yaoheng/test/log/54.log | grep "map at KnnAlgorithm.scala:507) finished in" | cut -d " " -f 14