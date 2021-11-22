#!bash
cd /home/yaoheng/test/spark-aknn
spark-submit --packages 'com.github.jelmerk:hnswlib-spark_2.3_2.11:0.0.49' test/maintest.py #--master local[4]