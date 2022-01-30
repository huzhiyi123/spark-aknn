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
from sklearn.cluster import KMeans as km
from tkinter import _flatten
import time


def hnsw_global_index_pddf(pddf,max_elements,dim,featurecol="features",partitionid_col="partition_id"):
    print("hnsw_global_index test")
    data = np.array(pddf[featurecol].values.tolist())
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

# 从queryVecList里面 heres
# queryVecList pd.dataframe labels:ndarrary
# 这里label 是找到的k近领向量的
def getMapCols(globaIndexDf,labels,partitionColName):
    res = []
    length = labels.shape[0]
    curdf=globaIndexDf[partitionColName]
    print("print(curdf[0]),curdf.shape",curdf.shape,type(curdf))
    for  i in range(length):
        currows = labels[i]
        curlist = (curdf.iloc[currows]).values.tolist()
        res.append(curlist)
    return res
"""
def getMapCols(globaIndexDf,labels,partitionColName):
    res = []
    length = labels.shape[0]
    for  i in range(length):
        currows = labels[i]
        curlist = (globaIndexDf.iloc[currows])[partitionColName].values.tolist()
        res.append(curlist)
    return res
"""

# custom function to sample rows within partitions
# df spark-df and return spark-df
# fraction 采样比例
# 从spark各个分区采样dataframe
def resample_in_partition(df, fraction, partition_col_name, seed=42):
      # create dictionary of sampling fractions per `partition_col_name`
  #df = sql_context.createDataFrame([tuple([1 + n]) for n in range(200)], ['number'])
  #https://coderedirect.com/questions/212771/stratified-sampling-with-pyspark
  fractions = df\
    .select(partition_col_name)\
    .distinct()\
    .withColumn('fraction', F.lit(fraction))\
    .rdd.collectAsMap()
  # stratified sampling
  print("fractions = df",fractions)
  sampled_df = df.stat.sampleBy(partition_col_name, fractions, seed)
  return sampled_df

#  spark-shell --master local-cluster[4,2] spark.default.parallelism = x * y
# 数据训练Kmeans 分区 分区的数据并没有归一化

## 加samplerate
def kmeansPartition(sc,sql_context,traindatapath,k,partitionColname,maxelement,traindatanum):
    traindata = fvecs_read(traindatapath)
    traindata1 = traindata[0:traindatanum]
    traindata = traindata.tolist()
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
def kmeansPandasDf(traindatapath,querydatapath,k=8,traindatanum=2000):
    data = fvecs_read(traindatapath)
    traindata = data[0:traindatanum]
    querydata = fvecs_read(querydatapath)
    l = len(querydata)
    df=pd.DataFrame(np.arange(l),columns=['id'])
    df['features']=querydata.tolist()
    kmeans = sklearn.cluster.KMeans(n_clusters=k, random_state=0).fit(traindata)
    res = kmeans.predict(querydata).reshape(l,1)
    df["partition"] = res.tolist()
    return df
"""
def kmeansPandasDf(data,k=8,traindatanum=2000):
    
    traindata = data[0:traindatanum]
    l = len(data)
    df=pd.DataFrame(np.arange(l),columns=['id'])
    df['features']=data.tolist()
    kmeans = km(n_clusters=k, random_state=0).fit(traindata)
    res = kmeans.predict(data).reshape(l,1).tolist()
    res=list(_flatten(res))
    df["partition"] = res
    return df

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
def processQueryVec(model,queryVec,globaIndexDf,queryPartitionsCol,\
    partitionCol="partitionCol",partitionnum=8,topkPartitionNum=4,knnQueryNum=10):
    T6 = time.time()
    labels, distances = model.knn_query(queryVec, k=knnQueryNum)
    T7 = time.time()
    globalserchtime=(T7-T6)*1000
    print("model.knn_query(queryVec, k=knnQueryNum) global index search time",globalserchtime)
    cols = getMapCols(globaIndexDf,labels,partitionCol)
    # unique 这些分区号 不足的填充其他分区 返回的是list
    cols = uniqueAndRefill(np.array(cols),topkPartitionNum,partitionnum)
    length = queryVec.shape[0]
    cur = pd.DataFrame(np.arange(length),columns=["id"])
    cur['features'] = queryVec.tolist()
    cur[queryPartitionsCol] = cols
    return cur,globalserchtime


def processSparkDfResult(result):
    #result_approximate = result.select('approximate')
    #tmp = result_approximate.select("approximate.neighbor")
    result_approximate = result.select('approximate.neighbor')
    l = result_approximate.toPandas().values.tolist()
    res=[]
    for i in range(len(l)):
        res.append(l[i][0])
    cur = np.array(res)
    return cur


def readDataSparkDf(sql_context,traindatapath):
        # 读取查询向量 并且全局索引查询预测的分区
    qd = fvecs_read(traindatapath)
    # id features(arrary) partioncol(int)
    vec = pd.DataFrame(np.arange(qd.shape[0]),columns=["id"])
    vec['normalized_features'] = qd.tolist()
    curschema = StructType([ StructField("id", IntegerType() ),StructField("normalized_features",ArrayType(DoubleType()))])
    df = sql_context.createDataFrame(vec,curschema)
    return df



def evaluate(index, xq, gt, k):
    nq = xq.shape[0]
    t0 = time.time()
    D, I = index.search(xq, k)  # noqa: E741
    t1 = time.time()

    recalls = {}
    i = 1
    while i <= k:
        recalls[i] = (I[:, :i] == gt[:, :1]).sum() / float(nq)
        i *= 10

    return (t1 - t0) * 1000.0 / nq, recalls


# predict groundtruth: np.arrary
def evaluatePredict(predict,groundtruth,k):
    l = predict.shape[0]
    real = groundtruth[:l,:k]
    cnt = 0 
    for i in range(l):
        cnt+=len(set(predict[i])&set(real[i]))
    print("l",l,"k",k)
    recall = cnt/float(l*k)
    #print("recall = cnt/float(l*k)",cnt," ","l",l,"k",k)
    return recall

# predict groundtruth: np.arrary
def evaluatePredictV2(predict,groundtruth,k):
    l = predict.shape[0]
    real = groundtruth[:l,:k]
    recall = np.mean(predict.reshape(-1)==groundtruth.reshape(-1))
    return recall