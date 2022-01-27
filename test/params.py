maxelement = 100000000
k=10
partitionnum=8
topkPartitionNum=3

sc = 1
m = int(50)
distanceFunction='cosine'
kmeanstrainrate=0.05

efConstruction=35
ef = int(4*22)
usesift=True

kmeanspath="/aknn/kmeans/"

gistlist=["gistpartition.csv","gistcentroids2.csv","gistcentroids1.csv"]  
siftlist=["siftpartition.csv","siftcentroids1.csv","siftcentroids2.csv"]

"""
datapath="/my/siftsmall/"
traindatapath=datapath+"siftsmall_base.fvecs"
querydatapath=datapath+"siftsmall_query.fvecs"
querygroundtruthpath=datapath+"siftsmall_groundtruth.ivecs"
"""
datapath="/data/"
traindatapath=datapath+"sift_base.fvecs"
querydatapath=datapath+"sift_query.fvecs"
querygroundtruthpath=datapath+"sift_groundtruth.ivecs"


gistpath="/data/mnist.hdf5"
# ef=10, efConstruction=200


