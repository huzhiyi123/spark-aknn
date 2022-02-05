cd /usr/spark-2.4.1
jarexample=spark-examples_2.11-2.4.1.jar
bin/spark-submit --master spark://9700147fade3:7077 --class org.apache.spark.examples.SparkPi examples/jars/$jarexample 1000
# bin/spark-submit --master spark://9700147fade3:7077 --class org.apache.spark.examples.SparkPi examples/jars/spark-examples_2.12-3.1.2.jar 1000
