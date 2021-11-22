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

