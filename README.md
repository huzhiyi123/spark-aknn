# spark-aknn
spark-aknn

. \
├── README.md \
├── aknn-main.py \
├── datasets.py      read data function \
└── sift_small_5MB.tar.gz data
## 设置token url的方法
### file .git/config 
```
[remote "origin"]
	url = https://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@github.com/huzhiyi123/spark-aknn.git \
	fetch = +refs/heads/*:refs/remotes/origin/*
```
1. 先删除远程url
2. git remote add origin https://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@github.com/huzhiyi123/spark-aknn.git
3. git push set upstream
   

local-cluster[N,cores,memory]
N模拟集群的 Slave（或worker）节点个数
cores模拟集群中各个Slave节点上的内核数
memory模拟集群的各个Slave节点上的内存大小
备注：参数之间没有空格，memory不能加单位
1、启动 Spark 伪分布式模式

spark-shell --master local-cluster[4,1,2048]



# recall
0.973
0.339

# log文件里 
这一条查找执行时间 map at KnnAlgorithm.scala:507) finished in

2021-11-30 10:11:57 INFO  DAGScheduler:54 - ShuffleMapStage 2 (map at KnnAlgorithm.scala:507) finished in 0.122 s


用法：bash run.sh logfilename &
run.sh : 测试主要函数 test/maintest.py
runtest.sh : 测试hnsw 与 bruteforce的搜索速度 test/test.py