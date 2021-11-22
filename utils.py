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

def hnsw_global_index_pddf(pddf,max_elements,dim,nf_col="normalized_features",partitionid_col="partition_id"):
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


# insert spark df into index
# return model and pandas df inserted into the index
# pandas df: col = nf_col partitionid_col
def hnsw_global_index_sparkdf(sparkdf,max_elements,dim,nf_col="normalized_features",partitionid_col="partition_id"):
    print("hnsw_global_index test")
    pddf = sparkdf.select(nf_col,partitionid_col).toPandas()
    data = np.array(pddf[nf_col].values.tolist())
    #print("np.array(pddf[nf_col].values.tolist()) data.shape",data.shape)
    p = hnswlib.Index(space='cosine', dim=dim)  # possible options are l2, cosine or ip
    p.init_index(max_elements=max_elements, ef_construction=100, M=16)
    # Controlling the recall by setting ef:
    # higher ef leads to better accuracy, but slower search
    p.set_ef(10)
    p.set_num_threads(4)  # by default using all available cores
    print("Adding first batch of %d elements" % (len(data)))
    p.add_items(data)
    return p,pddf

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
# 从spark各个分区采样dataframe
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

"""
把返回的最近领向量所在的区号的colist去重后再填充不同的区号到k
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
# globaIndexDf:pd df
def processQueryVec(model,queryVec,globaIndexDf,partitionIdColName,partitionnum=8,topkPartitionNum=3,knnQueryNum=10):
    labels, distances = model.knn_query(queryVec, k=knnQueryNum)
    cols = getMapCols(globaIndexDf,labels,partitionIdColName)
    # unique 这些分区号 不足的填充其他分区 返回的是list
    cols = uniqueAndRefill(np.array(cols),topkPartitionNum,partitionnum)
    length = queryVec.shape[0]
    cur = pd.DataFrame(np.arange(length),columns=["id"])
    cur['features'] = queryVec.tolist()
    cur[partitionIdColName] = cols
    return cur