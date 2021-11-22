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
from  data.datasets import fvecs_read,ivecs_read,fvecs_read_norm,ivecs_read_norm
from pyspark.mllib.clustering import KMeans,KMeansModel
import pyspark.sql.functions as F
import hnswlib
from sklearn.preprocessing import normalize
from hnsw_global import hnsw_global_index_sparkdf
from utils import *

path="/home/yaoheng/test/download/data.gz"

findspark.init() 
def func(data,row):
    a = np.array([[0,1,2],[1,2,3],[2,3,4]],dtype='int32')
    k=row
    d=np.zeros(k*3,dtype='int8').reshape(k,3)
    for i in range(k):
        d[i] = a[i%3] 
    tmp = pd.DataFrame(d)
    data = pd.concat([data,tmp], axis = 1)
    return data


def resample_in_partition(df, fraction, partition_col_name='partition_id', seed=42):
      # create dictionary of sampling fractions per `partition_col_name`
  #df = sql_context.createDataFrame([tuple([1 + n]) for n in range(200)], ['number'])
  df = df.withColumn('partition_id', F.spark_partition_id())
  fractions = df\
    .select(partition_col_name)\
    .distinct()\
    .withColumn('fraction', F.lit(fraction))\
    .rdd.collectAsMap()
  # stratified sampling
  sampled_df = df.stat.sampleBy(partition_col_name, fractions, seed)
  return sampled_df

"""
转化成pandas dataframe fit hnsw
sampledf schema
root
 |-- features: array (nullable = true)
 |    |-- element: double (containsNull = true)
 |-- id: integer (nullable = true)
 |-- partitionCol: integer (nullable = true)
 |-- normalized_features: array (nullable = true)
 |    |-- element: double (containsNull = false)
 |-- partition_id: integer (nullable = false)

"""

def hnsw_global_index(spark_df,nf_col="normalized_features",partitionid_col="partition_id"): # df = df.withColumn('partition_id', F.spark_partition_id())
    dim=128
    num_elements = spark_df.count()

    pddf = spark_df.select(nf_col,partitionid_col).toPandas()
    print("hnsw_global_index test")
    #dataframe  这一列转化
    print(pddf.columns)
    print(pddf.dtypes)
    print(pddf.index)
    data = pddf[nf_col].apply(lambda x: np.array(x))
    idx = pddf[partitionid_col]
    
    p = hnswlib.Index(space='cosine', dim=dim)  # possible options are l2, cosine or ip
    p.init_index(max_elements=num_elements, ef_construction=100, M=16)

    # Controlling the recall by setting ef:
    # higher ef leads to better accuracy, but slower search
    p.set_ef(10)

    p.set_num_threads(4)  # by default using all available cores
    print("Adding first batch of %d elements" % (len(data)))
    p.add_items(data)
    cur = data[num_elements-20:]
    # Query the elements for themselves and measure recall:
    labels, distances = p.knn_query(cur, k=8)
    print(labels)



# 数据训练Kmeans 分区
def kmeansPartition(sc,sql_context,traindatapath,k,partitionColname):
    traindata = fvecs_read(traindatapath).tolist()
    traindata_rdd = sc.parallelize(traindata)
    KMeans_fit = KMeans.train(traindata_rdd,k)
    #print("Final centers: " +str(i)+"____"+ str(KMeans_fit.clusterCenters))
    output = KMeans_fit.predict(traindata_rdd)
    traindata_rddi=traindata_rdd.zipWithIndex()
    outputi=output.zipWithIndex()
    curschema = StructType([StructField("features",ArrayType(DoubleType())),StructField("_id1", IntegerType() )])
    traindata_df = sql_context.createDataFrame(traindata_rddi,curschema)
    curschema = StructType([StructField(partitionColname, IntegerType()),StructField("_id2", IntegerType() )])
    outputi_df=sql_context.createDataFrame(outputi,curschema)
    res=traindata_df.join(outputi_df,traindata_df._id1==outputi_df._id2,"inner").drop("_id2")
    return res

def genRandomQueryCols(sc,sql_context,querydatapath,partionnum=4,row=100,col=3,querycolname="queryCols"):
    querydata = fvecs_read(querydatapath).tolist()
    querydata_rdd = sc.parallelize(querydata)
    querydata_rddi=querydata_rdd.zipWithIndex()
    
    curschema = StructType([StructField("features",ArrayType(DoubleType())),StructField("_id1", IntegerType() )])
    querydata_df = sql_context.createDataFrame(querydata_rddi,curschema)

    ar=np.random.randint(partionnum,size=(row,col)).tolist()
    partionnum_rdd = sc.parallelize(ar).zipWithIndex()
    curschema = StructType([StructField(querycolname,ArrayType(IntegerType())),StructField("_id2", IntegerType() )])
    partionnum_rdd_df = sql_context.createDataFrame(partionnum_rdd,curschema)
    
    res=querydata_df.join(partionnum_rdd_df,querydata_df._id1==partionnum_rdd_df._id2,"inner").drop("_id2")
    return res


def getStructType(doublenum=300,intnum=3):
    schema = [StructField("id", StringType())]
    for i in range(doublenum):
        cur = "_c" + str(i+1)
        schema.append(StructField(cur,DoubleType()))
    for i in range(intnum):
        cur = "_c" + str(doublenum+1+i)
        schema.append(StructField(cur,IntegerType()))
    return StructType(schema)

def test_normalizer(): #.set('spark.jars.packages', 'com.github.jelmerk:hnswlib-spark_2.3.0_2.11:0.0.50-SNAPSHOT')
    APP_NAME = "mytest" #setMaster("local[2]").
    conf = (SparkConf().setAppName(APP_NAME)).setSparkHome("/home/yaoheng/sparkhub/spark-2.3.0-bin-hadoop2.7")
    sc = SparkContext(conf=conf)
    sc.setCheckpointDir(gettempdir())
    sql_context = SQLContext(sc)
    traindatapath="/home/yaoheng/test/data/siftsmall/siftsmall_base.fvecs"
    querydatapath="/home/yaoheng/test/data/siftsmall/siftsmall_query.fvecs"
    partitioncolname="partitionCol"
    partitionnum=8
    nfcolname="normalized_features"

    # 分区并且 训练分布式hnsw
    words_df = kmeansPartition(sc,sql_context,traindatapath,partitionnum,partitioncolname)
    words_df.printSchema()
    normalizer = Normalizer(inputCol="features", outputCol=nfcolname)
    words_df=normalizer.transform(words_df)
    words_df.printSchema()
    words_df=words_df.withColumnRenamed('_id1','id')
    hnsw = HnswSimilarity(identifierCol='id', queryIdentifierCol='id',queryPartitionsCol='querypartitions',featuresCol='normalized_features', 
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
    hnsw_global_model,global_df = hnsw_global_index_sparkdf(sampledf,1000000,128,nf_col=nfcolname,partitionid_col=partitioncolname)
   
   
    # 读取查询向量 并且全局索引查询预测的分区
    qd = fvecs_read_norm(querydatapath)
    qd = qd[0:30]
    # id features(arrary) partioncol(int)
    queryvec = processQueryVec(hnsw_global_model,qd,sampledf,partitioncolname)

    curschema = StructType([ StructField("id", IntegerType() ),StructField("features",ArrayType(DoubleType())),StructField(partitioncolname,ArrayType(IntegerType()))])
    query_df = sql_context.createDataFrame(queryvec,curschema)
    query_df=query_df.withColumnRenamed('features','normalized_features')
    query_df.printSchema()
    
    result=model.transform(query_df)
    print(type(result),result.count())
    result.printSchema()
    #result.show()
    print(result.select('approximate').take(1))
    #pipeline = Pipeline(stages=[vector_assembler,vector_assembler2,normalizer,hnsw])
    sc.stop()
    print("hellow world pyyspark\n")



def test_normalizer_test(): #.set('spark.jars.packages', 'com.github.jelmerk:hnswlib-spark_2.3.0_2.11:0.0.50-SNAPSHOT')
    APP_NAME = "mytest" #setMaster("local[2]").
    conf = (SparkConf().setAppName(APP_NAME)).setSparkHome("/home/yaoheng/sparkhub/spark-2.3.0-bin-hadoop2.7")
    sc = SparkContext(conf=conf)
    sc.setCheckpointDir(gettempdir())
    sql_context = SQLContext(sc)
    currows=1000
    diamonds = pd.read_csv(path, nrows=currows,delimiter=' ',quotechar="\u0000")
    diamonds.rename(columns={',': 'id'},inplace = True)
    diamonds=func(diamonds,currows)
    column = diamonds.columns[0:]
    for i in range(303):
        curname = "_c" + str(i+1)
        diamonds.rename(columns={column[i+1]: curname},inplace = True)
    #print(diamonds.dtypes)
    words_df = sql_context.createDataFrame(diamonds,getStructType())
    words_df.printSchema()
# 把1行后面全部转化问向量
    vector_assembler = VectorAssembler(inputCols=words_df.columns[1:301], outputCol='features_as_vector')
    vector_assembler2 = VectorAssembler(inputCols=words_df.columns[301:304], outputCol='querypartitions')
    #words_df = vector_assembler2.transform(words_df)
    #print(words_df.head(2))
    normalizer = Normalizer(inputCol="features_as_vector", outputCol="normalized_features", outputType="vector")
# driver进行partition_col分区
# driver 进行采样构建全局索引 hnsw
# querypartition ,queryPartitionsCol='querypartitions'
    brute_force = BruteForceSimilarity(identifierCol='id', queryIdentifierCol='id',
                         featuresCol='normalized_features', distanceFunction='inner-product',
                         k=10, numPartitions=4, excludeSelf=True, similarityThreshold=0.4,
                         predictionCol='exact',outputFormat='minimal')
    hnsw = HnswSimilarity(identifierCol='id', queryIdentifierCol='id',queryPartitionsCol='querypartitions',featuresCol='normalized_features', 
                        distanceFunction='inner-product', m=16, ef=5, k=10, efConstruction=200, numPartitions=8, 
                        excludeSelf=True, predictionCol='approximate', outputFormat='minimal')
    hnsw.setPartitionCol('_c301')
    
    pipeline = Pipeline(stages=[vector_assembler,vector_assembler2,normalizer,hnsw])
    words_df_sample = words_df.sample(0.01)
    model = pipeline.fit(words_df)
    words_df_sample.printSchema()
    print(words_df_sample.head(1))
    #output = model.transform(words_df_sample)
    #evaluator = KnnSimilarityEvaluator(approximateNeighborsCol='approximate', exactNeighborsCol='exact')
    #accuracy = evaluator.evaluate(output)
    #print("accuracy:",accuracy)
    sc.stop()
    print("hellow world pyyspark\n")

def test_normalizer1():
    APP_NAME = "mytest" #setMaster("local[2]").
    conf = (SparkConf().setAppName(APP_NAME)).setSparkHome("/home/yaoheng/sparkhub/spark-2.3.0-bin-hadoop2.7")
    spark_context = SparkContext(conf=conf)
    spark_context.setCheckpointDir(gettempdir())

    sql_context = SQLContext(spark_context)

    df = sql_context.createDataFrame([
        [1, Vectors.dense([0.2, 0.9])],
        [2, Vectors.dense([0.2, 1.0])],
        [3, Vectors.dense([0.2, 0.1])],
    ], ['row_id', 'features'])
    df.groupBy('row_id')
    df2 = sql_context.createDataFrame([
        [1,0.2, 0.9],
        [2,0.2, 1.0],
        [3,0.2, 0.1],
    ], ['row_id', 'a','b'])
    df2.printSchema()
    vector_assembler = VectorAssembler(inputCols=df2.columns[1:], outputCol='features_as_vector')
    normalizer = Normalizer(inputCol="features_as_vector", outputCol="normalized_features")
    hnsw = HnswSimilarity(identifierCol='row_id', featuresCol='normalized_features', distanceFunction='cosine', m=32, ef=5, k=5,
                          efConstruction=200, numPartitions=100, excludeSelf=False, similarityThreshold=-1.0)
    pipeline = Pipeline(stages=[vector_assembler,normalizer,hnsw])
    model = pipeline.fit(df2)
    result = model.transform(df2)
    print("success")



def test_func():
    APP_NAME = "mytest" #setMaster("local[2]").
    conf = (SparkConf().setAppName(APP_NAME)).setSparkHome("/home/yaoheng/sparkhub/spark-2.3.0-bin-hadoop2.7")
    spark_context = SparkContext(conf=conf)
    spark_context.setCheckpointDir(gettempdir())
    sql_context = SQLContext(spark_context)     
    schema = [StructField("name", StringType()),StructField("abc", IntegerType()),StructField("efg", IntegerType())]
    for i in range(300):
        cur = "_c" + str(i)
        schema.append(StructField(cur, IntegerType()))
    curs = StructType(schema)
    df = sql_context.createDataFrame([["aa",5,1,5,3]],curs)
    



if __name__ == "__main__":
    test_normalizer()



# todo
"""
分区sample 
全局索引
"""
