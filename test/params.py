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
"""
datapath="/my/siftsmall/"
traindatapath=datapath+"siftsmall_base.fvecs"
querydatapath=datapath+"siftsmall_query.fvecs"
querygroundtruthpath=datapath+"siftsmall_groundtruth.ivecs"
"""
datapath="/my/sift/"
traindatapath=datapath+"sift_base.fvecs"
querydatapath=datapath+"sift_query.fvecs"
querygroundtruthpath=datapath+"sift_groundtruth.ivecs"


hdf5file="/workspace/gist/gist.hdf5"
# ef=10, efConstruction=200


def initparams():
    global maxelement,k,partitionnum,topkPartitionNum,ef,m,distanceFunction,kmeanstrainraten,efConstruction
    maxelement = 100000000
    k=10
    partitionnum=8
    topkPartitionNum=5
    
    sc = 1
    m = int(50)
    distanceFunction='cosine'
    kmeanstrainrate=0.05
    efConstruction=85
    ef = efConstruction
