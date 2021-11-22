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
from datasets import fvecs_read,ivecs_read,fvecs_read_norm,ivecs_read_norm
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
    partitioncolname="partitionCol"
    queryPartitionsCol='querypartitions'
    partitionnum=8
    nfcolname="normalized_features"

    # 分区并且 训练分布式hnsw
    words_df = kmeansPartition(sc,sql_context,traindatapath,partitionnum,partitioncolname)
    words_df.printSchema()
    normalizer = Normalizer(inputCol="features", outputCol=nfcolname)
    words_df=normalizer.transform(words_df)
    words_df.printSchema()
    words_df=words_df.withColumnRenamed('_id1','id')
    hnsw = HnswSimilarity(identifierCol='id', queryIdentifierCol='id',queryPartitionsCol=queryPartitionsCol,featuresCol='normalized_features', 
                        distanceFunction='inner-product', m=16, ef=5, k=10, efConstruction=200, numPartitions=8, 
                        excludeSelf=True, predictionCol='approximate', outputFormat='minimal')
    hnsw.setPartitionCol(partitioncolname)
    model=hnsw.fit(words_df)

    # 采样全部分区 训练全局索引
    sampledf = resample_in_partition(words_df,0.2)
    print("sampledf schema")
    sampledf.printSchema()
    print("sampledf.count()",sampledf.count())
    print("sampledf.take(1):",sampledf.take(1))
    sampledf_pandas = sampledf.toPandas()
    print(sampledf_pandas.columns)
    hnsw_global_model = hnsw_global_index_pddf(sampledf_pandas,1000000,128,nf_col=nfcolname,partitionid_col=partitioncolname)
   
    # 读取查询向量 并且全局索引查询预测的分区
    qd = fvecs_read_norm(querydatapath)
    qd = qd[0:30]
    # id features(arrary) partioncol(int)
    queryvec = processQueryVec(hnsw_global_model,qd,sampledf_pandas,partitioncolname)
    print("queryvec schema")
    print(queryvec.columns)
    curschema = StructType([ StructField("id", IntegerType() ),StructField("normalized_features",ArrayType(DoubleType())),StructField(queryPartitionsCol,ArrayType(IntegerType()))])
    query_df = sql_context.createDataFrame(queryvec,curschema)
    query_df.printSchema()
    
    result=model.transform(query_df)
    print(type(result),result.count())
    result.printSchema()
    #result.show()
    print(result.select('approximate').take(1))
    #pipeline = Pipeline(stages=[vector_assembler,vector_assembler2,normalizer,hnsw])
    sc.stop()
    print("hello world pyspark\n")


if __name__ == "__main__":
    testmain()


