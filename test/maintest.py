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
from tests import test_bruteforce,test_knn_evaluator,test_vector_converter,test_hnsw
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
sys.path.append("/home/yaoheng/test/spark-aknn")
from utils import *
from datasets import *
import time 
path="/home/yaoheng/test/download/data.gz"

findspark.init() 

traindatapath="/home/yaoheng/test/data/siftsmall/siftsmall_base.fvecs"
querydatapath="/home/yaoheng/test/data/siftsmall/siftsmall_query.fvecs"
querygroundtruthpath="/home/yaoheng/test/data/siftsmall/siftsmall_groundtruth.ivecs"
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
def SparkHnsw(): #.set('spark.jars.packages', 'com.github.jelmerk:hnswlib-spark_2.3.0_2.11:0.0.50-SNAPSHOT')
    APP_NAME = "mytest" #setMaster("local[2]").
    conf = (SparkConf().setAppName(APP_NAME)).setSparkHome("/home/yaoheng/sparkhub/spark-2.3.0-bin-hadoop2.7")
    sc = SparkContext(conf=conf)
    sc.setCheckpointDir(gettempdir())
    sql_context = SQLContext(sc)
    partitioncolname="partitionCol"
    queryPartitionsCol='querypartitions'
    nfcolname="normalized_features"
    # 分区并且 训练分布式hnsw 这里没有归一化
    words_df = kmeansPartition(sc,sql_context,traindatapath,partitionnum,partitioncolname,maxelement)
    # 这里上dataframe kmeans
    # RDD按照col分区
    # 
    """
    words_df.printSchema()
    """
    normalizer = Normalizer(inputCol="features", outputCol=nfcolname)
    #归一化后
    words_df=normalizer.transform(words_df)
    """
    words_df.printSchema()
    """
    words_df=words_df.withColumnRenamed('_id1','id')
    hnsw = HnswSimilarity(identifierCol='id', queryIdentifierCol='id',queryPartitionsCol=queryPartitionsCol,featuresCol='normalized_features', 
                        distanceFunction=distanceFunction, m=m, ef=ef, k=k, efConstruction=ef, numPartitions=partitionnum, 
                        excludeSelf=True, predictionCol='approximate', outputFormat='minimal')
    hnsw.setPartitionCol(partitioncolname)
    model=hnsw.fit(words_df)
    sampledf = resample_in_partition(words_df,0.05,partitioncolname)
    """
    # 采样全部分区 训练全局索引
    #kkk = words_df.select(partitioncolname).groupBy(partitioncolname).count()
    #kkk.collect()
    #print("print(kkk.collect())",kkk.collect())
    sampledf = resample_in_partition(words_df,0.05,partitioncolname)
    #jjj=sampledf.select(partitioncolname).groupBy(partitioncolname).count()
    #print("print(kkk.collect())",jjj.collect())

    # partition_id partitioncolname
    print("testdf=sampledf.select("",partitioncolname)")
    testdf=sampledf.select("partition_id",partitioncolname)
    #testdf.show(n=100)
    pddd=testdf.toPandas()
    print(pddd)
    """
    sampledf_pandas = sampledf.toPandas()
    #def hnsw_global_index_pddf(pddf,max_elements,dim,nf_col="normalized_features",partitionid_col="partition_id"):
    hnsw_global_model = hnsw_global_index_pddf(sampledf_pandas,1000000,128,nf_col=nfcolname,partitionid_col=partitioncolname)
    # 读取查询向量 并且全局索引查询预测的分区
    numpyquerydata = fvecs_read_norm(querydatapath)
    # id features(arrary) partioncol(int)
    T3 = time.time()
    queryvec = processQueryVec(hnsw_global_model,numpyquerydata,sampledf_pandas,partitioncolname,partitionnum=partitionnum,topkPartitionNum=3,knnQueryNum=10)
    T4 = time.time()
    curschema = StructType([ StructField("id", IntegerType() ),StructField("normalized_features",ArrayType(DoubleType())),StructField(queryPartitionsCol,ArrayType(IntegerType()))])
    query_df = sql_context.createDataFrame(queryvec,curschema)
    #query_df.printSchema()
    # todo with index
    T1 = time.time()
    print("start query")
    result=model.transform(query_df).orderBy("id")
    print("end query")
    T2 = time.time()
    predict = processSparkDfResult(result)
    timeUsed = (T2-T1)*1000
    print("timeUsed: ",timeUsed,"globalindextime",(T4-T3)*1000)
    print("predict",predict)
    
    
    groundtruth = ivecs_read(querygroundtruthpath)
    print("groundtruth[0:5]:",groundtruth[:,0:k])
    recall1 = evaluatePredict(predict,groundtruth,k)
    print("recall:",recall1)
    #if not openSparkUI:
    sc.stop()
    print("hello world SparkHnsw\n")
    return recall1


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
    querydf = readDataSparkDf(sql_context,querydatapath)
    bruteforce = BruteForceSimilarity(identifierCol='id', queryIdentifierCol='id', featuresCol='normalized_features',
                                    distanceFunction=distanceFunction, numPartitions=partitionnum, excludeSelf=False,
                                    predictionCol='approximate', outputFormat='minimal',k=k)
    model = bruteforce.fit(traindf)
    T1 = time.time()
    predict = model.transform(querydf).orderBy("id")
    T2 = time.time()
    timeUsed = (T2-T1)*1000
    print("timeUsed: ",timeUsed)
    groundtruth = ivecs_read(querygroundtruthpath)[:,:k]
    predict.printSchema()
    predict = processSparkDfResult(predict)
    recall = evaluatePredict(predict,groundtruth,k)
    print("recall",recall)
    print(predict)
    print(groundtruth)
    if not openSparkUI:
        sc.stop()
    print("hello world bruteForce\n")
    return recall


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
    """
    qd = fvecs_read_norm(querydatapath)
    # id features(arrary) partioncol(int)
    queryvec = pd.DataFrame(np.arange(qd.shape[0]),columns=["id"])
    queryvec['normalized_features'] = qd.tolist()
    curschema = StructType([ StructField("id", IntegerType() ),StructField("normalized_features",ArrayType(DoubleType()))])
    query_df = sql_context.createDataFrame(queryvec,curschema)
    """
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
    a=SparkHnsw()
    testmain_naiveSparkHnsw()
    b=bruteForce()
    #print(a,b)
    #testmain_naiveSparkHnsw()
    #bruteForce()
    #text = input("Please enter a text:")
    #test()

# cat /home/yaoheng/test/log/54.log | grep "map at KnnAlgorithm.scala:507) finished in" | cut -d " " -f 14