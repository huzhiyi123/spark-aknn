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
from kmeans_repartition_utils import *
import time 
from time import sleep
findspark.init() 


def testdoublekmeansHnsw(): #.set('spark.jars.packages', 'com.github.jelmerk:hnswlib-spark_2.3.0_2.11:0.0.50-SNAPSHOT')
    print("/my/spark-aknn/test/testdoublekmeans.py")
    APP_NAME = "mytest" #setMaster("local[2]").
    conf = (SparkConf().setAppName(APP_NAME))#.setSparkHome("/home/yaoheng/sparkhub/spark-2.3.0-bin-hadoop2.7")
    sc = SparkContext(conf=conf)
    sc.setCheckpointDir(gettempdir())
    sql_context = SQLContext(sc)
    partitioncolname="partition"
    queryPartitionsCol='querypartitions'
    nfcolname="normalized_features"
    featuresCol="features"
    partitionreal = "mappartition"
    # 分区并且 训练分布式hnsw 这里没有归一化
    traindata = fvecs_read(traindatapath)   #.reshape(-1,128)  #[0,base*num:-1]
    print(type(traindata),traindata.shape)
    # 2 4 6 8
    datalen=len(traindata)

    partitionnumreal=partitionnum
    partitionnummap=int(partitionnum*10)
    
    T1 = time.time()
    df,centroids1,centroids2=kmeansPandasDfV2(traindata,k1=partitionnumreal,k2=partitionnummap,\
        traindatanum=int(datalen*kmeanstrainrate),partitioncolname=partitioncolname)
    # df 的 partition col映射到真是的cols
    # df=pd.DataFrame(columns=['id','features','partition'])#(np.arange(l),columns=['id'])
    T2 = time.time()
    kmeanspartitiontime=(T2-T1)*1000
    print("kmeanspartitiontime",kmeanspartitiontime)

    allpartitionrank=getallpartitionrank(centroids1,centroids2,partitionnumreal,partitionnummap)
    eachpartitonnum=geteachpartitonnum(df)
    repartitionres,repartitionnum=repartition(allpartitionrank,eachpartitonnum,partitionnumreal,partitionnummap,df.shape[0])
    sampledf_pandas=getsampledata(df,samplerate=0.05)
    partitionmap = getrepartitionmap(repartitionres,partitionnumreal,partitionnummap)
    print("partitionmap",partitionmap)

    #df=pd.DataFrame(columns=['id','features','partition'])#(np.arange(l),columns=['id'])
    # df的 partition转化成新的partition
    def mapx(x):
        x = partitionmap[x]
        return x
    df[partitionreal]=df[partitioncolname].apply(lambda x:mapx(x))
    print(df[0:20])


    # id features partition
    curschema = StructType([StructField("id", IntegerType()),
    StructField("features",ArrayType(DoubleType())),
    StructField(partitioncolname, IntegerType() ),
    StructField(partitionreal, IntegerType() )
    ])
    words_df = sql_context.createDataFrame(df,curschema)
    #words_df.show()
    #,queryPartitionsCol=queryPartitionsCol
    hnsw = HnswSimilarity(identifierCol='id',queryIdentifierCol='id',featuresCol=featuresCol, 
                        distanceFunction=distanceFunction, m=m, ef=ef, k=k, efConstruction=ef, numPartitions=partitionnum, 
                        excludeSelf=True, predictionCol='approximate', outputFormat='minimal')
    hnsw.setPartitionCol(partitionreal)
    T3 = time.time()
    #print("word_df.count()",words_df.count())
    model=hnsw.fit(words_df)
    T4 = time.time()
    localindexconstructtime=(T4-T3)*1000
    print("localindexconstructtime",localindexconstructtime)
    print("model=hnsw.fit(words_df) done")
    T5 = time.time()
    #def hnsw_global_index_pddf(pddf,max_elements,dim,nf_col="normalized_features",partitionid_col="partition_id"):
    hnsw_global_model=hnsw_global_index_wrapper(sampledf_pandas,1000000,128,featurecol='features')
    T6 = time.time()
    globalindexconstructtime=(T6-T5)*1000
    print("globalindexconstructtime",globalindexconstructtime)

    # 读取查询向量 并且全局索引查询预测的分区
    numpyquerydata = fvecs_read(querydatapath)
    numpyquerydata = numpyquerydata
    # id features(arrary) partioncol(int)

    # 这里处理
    # sampledf_pandas: 'id','features','partition'
    knnQueryNum=30
    queryvec,globalsearchtime = processQueryVecv2(hnsw_global_model,numpyquerydata,\
    sampledf_pandas,queryPartitionsCol,partitioncolname,partitionmap,\
    partitionnum=partitionnum,topkPartitionNum=topkPartitionNum,knnQueryNum=knnQueryNum)
    print("globalsearchtime",globalsearchtime)


    curschema = StructType([ StructField("id", IntegerType() ),\
        StructField(featuresCol,ArrayType(DoubleType())),
        StructField(queryPartitionsCol,ArrayType(IntegerType()))])
    query_df = sql_context.createDataFrame(queryvec,curschema)
    #query_df.printSchema()
    # todo with index
    T7 = time.time()
    print("start query")
    result=model.transform(query_df).orderBy("id")
    print("result.count()",result.count())
    print("end query")
    T8 = time.time()
    localsearchtime=(T8-T7)*1000
    print("localsearchtime",localsearchtime)

    predict = processSparkDfResult(result)
    print("predict = processSparkDfResult(result)",predict)

    groundtruth = ivecs_read(querygroundtruthpath)
    #print("groundtruth[0:5]:",groundtruth[:,0:k])
    if(len(predict) != 0):
        recall1 = evaluatePredict(predict,groundtruth,k)
        print("recall:",recall1)
    #if not openSparkUI:
    #sleep(150)
    sc.stop()
    #input("ffew")
    totalsearchtime = localsearchtime + globalsearchtime
    totalconstructtime=kmeanspartitiontime+localindexconstructtime+globalindexconstructtime
    
    print("totalconstructtime",totalconstructtime,\
        "localsearchtime",localsearchtime,"globalsearchtime",globalsearchtime)

    print("totalsearchtime",totalsearchtime,"localsearchtime",localsearchtime,"globalsearchtime",globalsearchtime)
    print("hello world testdoublekmeansHnsw\n")
    #return recall1

if __name__ == "__main__":
    testdoublekmeansHnsw()
# k 的对比 ef