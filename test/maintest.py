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

from pyspark.mllib.clustering import KMeans,KMeansModel
import pyspark.sql.functions as F
import hnswlib
from sklearn.preprocessing import normalize
import sys
sys.path.append("/home/yaoheng/test/spark-aknn")
from utils import *
from datasets import *
path="/home/yaoheng/test/download/data.gz"

findspark.init() 

def testmain(): #.set('spark.jars.packages', 'com.github.jelmerk:hnswlib-spark_2.3.0_2.11:0.0.50-SNAPSHOT')
    APP_NAME = "mytest" #setMaster("local[2]").
    conf = (SparkConf().setAppName(APP_NAME)).setSparkHome("/home/yaoheng/sparkhub/spark-2.3.0-bin-hadoop2.7")
    sc = SparkContext(conf=conf)
    sc.setCheckpointDir(gettempdir())
    sql_context = SQLContext(sc)
    traindatapath="/home/yaoheng/test/data/siftsmall/siftsmall_base.fvecs"
    querydatapath="/home/yaoheng/test/data/siftsmall/siftsmall_query.fvecs"
    querygroundtruthpath="/home/yaoheng/test/data/siftsmall/siftsmall_groundtruth.ivecs"
    partitioncolname="partitionCol"
    queryPartitionsCol='querypartitions'
    partitionnum=8
    nfcolname="normalized_features"
    k=10
    # 分区并且 训练分布式hnsw 这里没有归一化
    words_df = kmeansPartition(sc,sql_context,traindatapath,partitionnum,partitioncolname)
    words_df.printSchema()

    normalizer = Normalizer(inputCol="features", outputCol=nfcolname)
    #归一化后
    words_df=normalizer.transform(words_df)
    words_df.printSchema()
    words_df=words_df.withColumnRenamed('_id1','id')
    hnsw = HnswSimilarity(identifierCol='id', queryIdentifierCol='id',queryPartitionsCol=queryPartitionsCol,featuresCol='normalized_features', 
                        distanceFunction='inner-product', m=16, ef=5, k=k, efConstruction=200, numPartitions=8, 
                        excludeSelf=True, predictionCol='approximate', outputFormat='minimal')
    hnsw.setPartitionCol(partitioncolname)
    model=hnsw.fit(words_df)

    # 采样全部分区 训练全局索引
    sampledf = resample_in_partition(words_df,0.2)

    sampledf_pandas = sampledf.toPandas()
    hnsw_global_model = hnsw_global_index_pddf(sampledf_pandas,1000000,128,nf_col=nfcolname,partitionid_col=partitioncolname)
   
    # 读取查询向量 并且全局索引查询预测的分区
    qd = fvecs_read_norm(querydatapath)
    # id features(arrary) partioncol(int)
    queryvec = processQueryVec(hnsw_global_model,qd,sampledf_pandas,partitioncolname)

    curschema = StructType([ StructField("id", IntegerType() ),StructField("normalized_features",ArrayType(DoubleType())),StructField(queryPartitionsCol,ArrayType(IntegerType()))])
    query_df = sql_context.createDataFrame(queryvec,curschema)
    query_df.printSchema()
    # todo with index
    result=model.transform(query_df).orderBy("id")
    predict = processSparkDfResult(result)

    print("predict",predict)
    groundtruth = ivecs_read(querygroundtruthpath)
    print("groundtruth[0:5]:",groundtruth[0:5,0:10])
    res,cnt,recall = evaluatePredict(predict,groundtruth,k)
    print("recall:",recall)
    print("res\n",res)
    print("cnt:",cnt)
    sc.stop()
    print("hello world pyspark\n")

def readtraindata(sc,sql_context,traindatapath,k,partitionColname):
    traindata = fvecs_read(traindatapath).tolist()
    traindata_rdd = sc.parallelize(traindata)
    traindata_rddi=traindata_rdd.zipWithIndex()
    curschema = StructType([StructField("features",ArrayType(DoubleType())),StructField("id", IntegerType() )])
    traindata_df = sql_context.createDataFrame(traindata_rddi,curschema)
    return traindata_df


def testmain_naiveSparkHnsw(): #.set('spark.jars.packages', 'com.github.jelmerk:hnswlib-spark_2.3.0_2.11:0.0.50-SNAPSHOT')
    APP_NAME = "mytest" #setMaster("local[2]").
    conf = (SparkConf().setAppName(APP_NAME)).setSparkHome("/home/yaoheng/sparkhub/spark-2.3.0-bin-hadoop2.7")
    sc = SparkContext(conf=conf)
    sc.setCheckpointDir(gettempdir())
    sql_context = SQLContext(sc)
    traindatapath="/home/yaoheng/test/data/siftsmall/siftsmall_base.fvecs"
    querydatapath="/home/yaoheng/test/data/siftsmall/siftsmall_query.fvecs"
    querygroundtruthpath="/home/yaoheng/test/data/siftsmall/siftsmall_groundtruth.ivecs"
    partitioncolname="partitionCol"
    queryPartitionsCol='querypartitions'
    partitionnum=8
    nfcolname="normalized_features"
    k=10
    # 分区并且 训练分布式hnsw 这里没有归一化
    words_df = readtraindata(sc,sql_context,traindatapath,partitionnum,partitioncolname)
    words_df.printSchema()

    normalizer = Normalizer(inputCol="features", outputCol=nfcolname)
    #归一化后
    words_df=normalizer.transform(words_df)
    words_df.printSchema()
    hnsw = HnswSimilarity(identifierCol='id', queryIdentifierCol='id',queryPartitionsCol=None,featuresCol='normalized_features', 
                        distanceFunction='inner-product', m=16, ef=5, k=k, efConstruction=200, numPartitions=8, 
                        excludeSelf=True, predictionCol='approximate', outputFormat='minimal')
    #hnsw.setPartitionCol(partitioncolname)
    model=hnsw.fit(words_df)

    # 采样全部分区 训练全局索引
    sampledf = resample_in_partition(words_df,0.2)

    sampledf_pandas = sampledf.toPandas()
    hnsw_global_model = hnsw_global_index_pddf(sampledf_pandas,1000000,128,nf_col=nfcolname,partitionid_col=partitioncolname)
   
    # 读取查询向量 并且全局索引查询预测的分区
    qd = fvecs_read_norm(querydatapath)
    # id features(arrary) partioncol(int)
    queryvec = pd.DataFrame(np.arange(qd.shape[0]),columns=["id"])
    queryvec['features'] = qd.tolist()
    curschema = StructType([ StructField("id", IntegerType() ),StructField("normalized_features",ArrayType(DoubleType()))])
    query_df = sql_context.createDataFrame(queryvec,curschema)
    query_df.printSchema()
    # todo with index
    result=model.transform(query_df).orderBy("id")
    predict = processSparkDfResult(result)

    print("predict",predict)
    groundtruth = ivecs_read(querygroundtruthpath)
    print("groundtruth[0:5]:",groundtruth[0:5,0:10])
    res,cnt,recall = evaluatePredict(predict,groundtruth,k)
    print("recall:",recall)
    print("res\n",res)
    print("cnt:",cnt)
    sc.stop()
    print("hello world pyspark\n")






if __name__ == "__main__":
    #testmain()
    testmain_naiveSparkHnsw()
