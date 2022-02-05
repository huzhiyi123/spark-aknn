#!bash
logfold="/my/log/"
aknnpath=/my/spark-aknn
package=com.github.jelmerk:hnswlib-spark_2.4_2.11:1.0.0 #com.github.jelmerk:hnswlib-spark_2.4_2.12:1.0.0


function function_name {
    cd $aknnpath
    filename=$1
    abpath=$logfold$filename
    echo $abpath
    spark-submit --master spark://9700147fade3:7077 --packages $package test/maintest.py  > $abpath #--master local[32]
    cat test/params.py >> $abpath
    cat $abpath | grep recall >> $abpath
    #cat $abpath | grep "map at KnnAlgorithm.scala:507) finished in" >> $abpath
}

function_name $1

echo $aknnpath

echo $package