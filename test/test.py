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
dim = 128
rate= 40
num_elements = int(rate*10000) #
#10000
# Generating sample data
data = np.float32(np.random.random((num_elements, dim)))
query = np.float32(np.random.random((int(num_elements/100), dim)))
query2 = np.float32(np.random.random((int(num_elements/100), dim)))

def getdf(sql_context,qd):
        # 读取查询向量 并且全局索引查询预测的分区
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
    # 分区并且 训练分布式hnsw 这里没有归一化
    traindf = getdf(sql_context,data)
    querydf = getdf(sql_context,query)
    bruteforce = BruteForceSimilarity(identifierCol='id', queryIdentifierCol='id', featuresCol='normalized_features',
                                    distanceFunction=distanceFunction, numPartitions=partitionnum, excludeSelf=False,
                                    predictionCol='approximate', outputFormat='minimal',k=k)
    model = bruteforce.fit(traindf)
    T1 = time.time()
    predict = model.transform(querydf).orderBy("id")
    T2 = time.time()
    sc.stop()
    #predict.show()
    print("bruteForce timeUsed\n",(T2-T1)*1000)

def testmain_naiveSparkHnsw(): #.set('spark.jars.packages', 'com.github.jelmerk:hnswlib-spark_2.3.0_2.11:0.0.50-SNAPSHOT')
    APP_NAME = "mytest" #setMaster("local[2]").
    conf = (SparkConf().setAppName(APP_NAME)).setSparkHome("/home/yaoheng/sparkhub/spark-2.3.0-bin-hadoop2.7")
    sc = SparkContext(conf=conf)
    sc.setCheckpointDir(gettempdir())
    sql_context = SQLContext(sc)
    # 分区并且 训练分布式hnsw 这里没有归一化
    words_df = getdf(sql_context,data)
    query_df = getdf(sql_context,query)
    #query_df2 = getdf(sql_context,query2)
    hnsw = HnswSimilarity(identifierCol='id', queryIdentifierCol='id',featuresCol='normalized_features',
                         distanceFunction=distanceFunction,m=m, ef=ef, k=k,efConstruction=ef,
                         numPartitions=partitionnum,  predictionCol='approximate',excludeSelf=True)
    model=hnsw.fit(words_df)
    T1 = time.time()
    result=model.transform(query_df)
    T2 = time.time()
    timeUsed = (T2-T1)*1000
    #result.show()
    sc.stop()
    print("testmain_naiveSparkHnsw timeUsed\n",timeUsed)

def kmeansPartition1(sc,sql_context,traindata,k,partitionColname,maxelement):
    if(traindata.shape[0]>maxelement):
        traindata = traindata[:maxelement]
    traindata = traindata.tolist()
    traindata_rdd = sc.parallelize(traindata)
    KMeans_fit = KMeans.train(traindata_rdd,k)
    #print("Final centers: " +str(i)+"____"+ str(KMeans_fit.clusterCenters))
    output = KMeans_fit.predict(traindata_rdd)
    traindata_rddi=traindata_rdd.zipWithIndex()
    outputi=output.zipWithIndex()
    curschema = StructType([StructField("normalized_features",ArrayType(DoubleType())),StructField("_id1", IntegerType() )])
    traindata_df = sql_context.createDataFrame(traindata_rddi,curschema)
    curschema = StructType([StructField(partitionColname, IntegerType()),StructField("_id2", IntegerType() )])
    outputi_df=sql_context.createDataFrame(outputi,curschema)
    res=traindata_df.join(outputi_df,traindata_df._id1==outputi_df._id2,"inner").drop("_id2")
    return res



def SparkHnsw(): #.set('spark.jars.packages', 'com.github.jelmerk:hnswlib-spark_2.3.0_2.11:0.0.50-SNAPSHOT')
    APP_NAME = "mytest" #setMaster("local[2]").
    conf = (SparkConf().setAppName(APP_NAME)).setSparkHome("/home/yaoheng/sparkhub/spark-2.3.0-bin-hadoop2.7")
    sc = SparkContext(conf=conf)
    sc.setCheckpointDir(gettempdir())
    sql_context = SQLContext(sc)
    partitioncolname="partitionCol"
    queryPartitionsCol='querypartitions'
    nfcolname="normalized_features"

    words_df = kmeansPartition1(sc,sql_context,data,partitionnum,partitioncolname,maxelement)
    #normalizer = Normalizer(inputCol="features", outputCol=nfcolname)
    #words_df=normalizer.transform(words_df)
    words_df=words_df.withColumnRenamed('_id1','id')
    hnsw = HnswSimilarity(identifierCol='id', queryIdentifierCol='id',queryPartitionsCol=queryPartitionsCol,featuresCol='normalized_features', 
                        distanceFunction=distanceFunction, m=m, ef=ef, k=k, efConstruction=ef, numPartitions=partitionnum, 
                        excludeSelf=True, predictionCol='approximate', outputFormat='minimal')
    hnsw.setPartitionCol(partitioncolname)
    model=hnsw.fit(words_df)
    sampledf = resample_in_partition(words_df,0.05,partitioncolname)
    sampledf_pandas = sampledf.toPandas()
    #def hnsw_global_index_pddf(pddf,max_elements,dim,nf_col="normalized_features",partitionid_col="partition_id"):
    hnsw_global_model = hnsw_global_index_pddf(sampledf_pandas,1000000,128,nf_col=nfcolname,partitionid_col=partitioncolname)
    # 读取查询向量 并且全局索引查询预测的分区

    # id features(arrary) partioncol(int)
    T3 = time.time()
    queryvec = processQueryVec(hnsw_global_model,query,sampledf_pandas,partitioncolname,partitionnum=partitionnum,topkPartitionNum=3,knnQueryNum=10)
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
    sc.stop()
    print("SparkHnsw timeUsed: ",timeUsed,"globalindextime",(T4-T3)*1000)
    print("predict",predict)
    



bruteForce()
testmain_naiveSparkHnsw()
SparkHnsw()
#text = input("Please enter a text:")