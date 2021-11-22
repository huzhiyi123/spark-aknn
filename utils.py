import pandas as pd
# coding=utf-8
from pyspark_hnsw.linalg import Normalizer
from pyspark_hnsw.knn import HnswSimilarity
from pyspark.ml.linalg import Vectors
from pyspark.sql import SQLContext, column
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
import numpy as np
import hnswlib
import pyspark.sql.functions as F
from datasets import fvecs_read,ivecs_read,fvecs_read_norm,ivecs_read_norm
from pyspark.mllib.clustering import KMeans,KMeansModel
import pyspark.sql.functions as F
from pyspark_hnsw.evaluation import KnnSimilarityEvaluator
from pyspark.ml.linalg import Vectors
from pyspark.sql import SQLContext,DataFrame
from pyspark.sql.types import *


def hnsw_global_index(pddf,max_elements,dim,nf_col="normalized_features",partitionid_col="partition_id"):
    print("hnsw_global_index test")
    data = np.array(pddf[nf_col].values.tolist())
    p = hnswlib.Index(space='cosine', dim=dim)  # possible options are l2, cosine or ip
    p.init_index(max_elements=max_elements, ef_construction=100, M=16)

    # Controlling the recall by setting ef:
    # higher ef leads to better accuracy, but slower search
    p.set_ef(10)
    p.set_num_threads(4)  # by default using all available cores
    print("Adding first batch of %d elements" % (len(data)))
    p.add_items(data)
    return p

# 从queryVecList里面
# queryVecList pd.dataframe labels:ndarrary
def getMapCols(queryVecList,labels,partitionColName):
    res = []
    length = labels.shape[0]
    for  i in range(length):
        currows = labels[i]
        curlist = (queryVecList.iloc[currows])[partitionColName].values.tolist()
        res.append(curlist)
    return res


# custom function to sample rows within partitions
# df spark-df and return spark-df
# fraction 采样比例
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


# 数据训练Kmeans 分区 分区的数据并没有归一化
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

"""
ar numpy ndarray
k 需要hnsw扫描的不同分区总数
partitionnum 总数
"""
def uniqueAndRefill(ar,k=3,partitionnum=8):
    res=[]
    for i in range(ar.shape[0]):
        cur=np.unique(ar[i]).tolist()
        l = len(cur)
        if(l>k):
            cur = cur[0:k]
        if(l<k):
            curmax=0
            flags = np.zeros(partitionnum)
            for j in range(l):
                flags[cur[j]] = 1
            idx=0
            for j in range(k-l):
                while idx < partitionnum:
                    if(flags[idx]==0):
                        cur.append(idx)
                        flags[idx]=1
                        idx+=1
                        break
                    idx+=1
        res.append(cur)
    return res

# queryVec是np arrary 返回含有query partition的df
# knnQueryNum knn选取最近的knnQueryNum向量 然后找到最近向量所在的分区（topkPartitionNum个）
# 默认的分区总部概述是partitionnum
# df: id features partitionIdColName
def processQueryVec(model,queryVec,globaIndexDf,partitionIdColName,partitionnum=8,topkPartitionNum=3,knnQueryNum=10):
    labels, distances = model.knn_query(queryVec, k=knnQueryNum)
    cols = getMapCols(globaIndexDf,labels,partitionIdColName)
    # unique 这些分区号 不足的填充其他分区 返回的是list
    cols = uniqueAndRefill(np.array(cols),topkPartitionNum,partitionnum)
    cur = pd.DataFrame(np.arange(queryVec.shape[0]),"id")
    cur['features'] = queryVec.tolist()
    cur[partitionIdColName] = cols
    return cur


def ggf(df):
    curschema = StructType([StructField("features",ArrayType(DoubleType())),StructField("_id1", IntegerType() )])

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

"""