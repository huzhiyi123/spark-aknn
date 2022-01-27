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
aknnfold="/aknn/"
sys.path.append(aknnfold)
sys.path.append(aknnfold+"test")
from utils import *
from datasets import *
from kmeans_repartition_utils import *
from maintest import *

import time 
from time import sleep

import h5py
import numpy as np


findspark.init() 

traindata = 0
numpyquerydata = 0
groundtruth = 0

gistpath="/data/mnist.hdf5"


rate = 5

initparams()
print("maxelement,k,partitionnum,topkPartitionNum,ef,m,distanceFunction,kmeanstrainrate,efConstruction,usesift\n",
maxelement,k,partitionnum,topkPartitionNum,ef,m,distanceFunction,kmeanstrainrate,efConstruction,usesift)
#usesift = False
efConstructionlist = [15,20,50,100,150]
efConstruction=100000
initparams()
print(efConstruction)