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
import time 
from sklearn.cluster import KMeans as km
from tkinter import _flatten
findspark.init() 

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

datapath="/my/siftsmall/"
traindatapath=datapath+"siftsmall_base.fvecs"
querydatapath=datapath+"siftsmall_query.fvecs"
querygroundtruthpath=datapath+"siftsmall_groundtruth.ivecs"


traindata = fvecs_read(traindatapath) #s.reshape(-1,128)  #[0,base*num:-1]
print(type(traindata),traindata.shape)

a=kmeansPandasDf(traindata,k=8,traindatanum=500)
print(a.shape)


