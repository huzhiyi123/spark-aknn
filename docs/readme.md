40W数据5Gdrivermemory——naivehnswVSbruteforce.log
bruteVSnaiveHNSW.log
bruteforceVSkmeanshnsw.log
好的实验结果.log 



'''
40W条数据

maxelement = 100000000
k=10
partitionnum=8
ef = int(1.5*k)
sc = 1
m = int(30)
distanceFunction='euclidean'

bruteForce timeUsed
 438.96484375
testmain_naiveSparkHnsw timeUsed
 247.14422225952148
SparkHnsw timeUsed:  143.02992820739746 globalindextime 697.6580619812012


bruteForce timeUsed
 330.949068069458
testmain_naiveSparkHnsw timeUsed 139.93191719055176
SparkHnsw timeUsed:  121.39582633972168 globalindextime 636.0459327697754



'''