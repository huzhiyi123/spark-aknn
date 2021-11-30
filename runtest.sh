#!bash
logfold="/home/yaoheng/test/log/"


function function_name {
    cd /home/yaoheng/test/spark-aknn
    filename=$1
    abpath=$logfold$filename
    echo $abpath
    spark-submit --packages 'com.github.jelmerk:hnswlib-spark_2.3_2.11:0.0.49' test/test.py --driver-memory 6G local-cluster[4,1,2048] > $abpath #--master local[32]
    cat test/params.py >> $abpath
    cat $abpath | grep recall >> $abpath
    cat $abpath | grep "map at KnnAlgorithm.scala:507) finished in" >> $abpath
}

function_name $1
