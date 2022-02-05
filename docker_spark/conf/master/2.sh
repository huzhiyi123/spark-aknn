#!bash
logfold="/my/log/"
aknnpath=/my/spark-aknn
package=com.github.jelmerk:hnswlib-spark_2.4_2.11:1.0.0 #com.github.jelmerk:hnswlib-spark_2.4_2.12:1.0.0
  #spark://b53cb1e828da:7077 \ --master local-cluster[8,1,1024] \  local[8]
# 90c8cd91fc05
function function_name {
    cd $aknnpath
    filename=$1
    abpath=$logfold$filename
    echo $abpath
    nohup spark-submit \
    --master spark://master:7077 \
    --conf spark.driver.maxResultSize=2G \
    --conf spark.scheduler.listenerbus.eventqueue.capacity=100000 \
     --conf spark.shuffle.service.enabled=true \
    --packages $package --driver-memory 10G test/maintest.py > $abpath 
    #--num-executors 8 --executor-memory 1G 
    #--master local[32]
    # --master spark://90c8cd91fc05:7077 \
    cat test/params.py >> $abpath
    cat $abpath | grep recall >> $abpath
    #cat $abpath | grep "map at KnnAlgorithm.scala:507) finished in" >> $abpath
}

#--conf kafka.version=0.10  
#--executor-cores 2  
#--executor-memory 2048M 
#--driver-memory 512M 
#--num-executors 1 


function_name $1

echo $aknnpath

echo $package


