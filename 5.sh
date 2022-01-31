#!bash
source env.sh  
#spark://b53cb1e828da:7077 \ --master local-cluster[8,1,1024] \  local[8] --master spark://master:7077
# 90c8cd91fc05
totalcores=0
executorcores=0
function function_name {
    cd $aknnpath
    filename=$1
    abpath=$logfold$filename
    echo $abpath
    nohup spark-submit \
    --master spark://master:7077 \
    --packages $package --driver-memory 5G  --executor-memory 2G  --executor-cores $executorcores \
    --conf spark.rpc.message.maxSize=1024 --total-executor-cores $totalcores \
    test/testdoublekmeans.py >> $abpath 
    #--num-executors 8 --executor-memory 1G 
    #--master local[32]
    # --master spark://90c8cd91fc05:7077 \
    cat test/params.py >> $abpath
    cat $abpath | grep recall >> $abpath
    #cat $abpath | grep "map at KnnAlgorithm.scala:507) finished in" >> $abpath
}
# --conf spark.driver.maxResultSize=4G \
#    --conf spark.scheduler.listenerbus.eventqueue.capacity=100000 \
#     --conf spark.shuffle.service.enabled=true \
#--conf kafka.version=0.10  
#--executor-cores 2  
#--executor-memory 2048M 
#--driver-memory 512M 
#--num-executors 1 

pyfile=0
function function_name2 {
    cd $aknnpath
    filename=$1
    abpath=$logfold$filename
    echo $abpath
    nohup spark-submit \
    --master spark://master:7077 \
    --packages $package --driver-memory 5G  --executor-memory 2G  --executor-cores $totalcores \
    --conf spark.rpc.message.maxSize=1024 --total-executor-cores 10 \
    test/$pyfile >> $abpath 
    #--num-executors 8 --executor-memory 1G 
    #--master local[32]
    # --master spark://90c8cd91fc05:7077 \
    cat test/params.py >> $abpath
    cat $abpath | grep recall >> $abpath
    #cat $abpath | grep "map at KnnAlgorithm.scala:507) finished in" >> $abpath
}

totalcores=8
executorcores=2
pyfile=testdoublekmeans.py
function_name $1


echo $aknnpath

echo $package


