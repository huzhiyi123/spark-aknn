import pyspark.sql.functions as F
from pyspark.sql import SparkSession
from pyspark import SparkContext, SparkConf
from tempfile import gettempdir
from pyspark.sql import SQLContext,DataFrame
APP_NAME = "mytest" #setMaster("local[2]").
conf = (SparkConf().setAppName(APP_NAME)).setSparkHome("/home/yaoheng/sparkhub/spark-2.3.0-bin-hadoop2.7")
spark_context = SparkContext(conf=conf)
spark_context.setCheckpointDir(gettempdir())

sql_context = SQLContext(spark_context)

# create example dataframe with numbers from 1 to 100
df = sql_context.createDataFrame([tuple([1 + n]) for n in range(2000)], ['number'])
print("df.rdd.getNumPartitions()",df.rdd.getNumPartitions())   # => 8
print(df.count())
# custom function to sample rows within partitions
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

df = resample_in_partition(df, fraction=0.2)
l = df.collect()
print(type(df))
df.show()
df.printSchema()
print("allpartition count=",df.count())
print(l)





