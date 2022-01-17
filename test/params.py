maxelement = 100000000
k=10
partitionnum=8
topkPartitionNum=3
ef = int(3*k)
sc = 1
m = int(25)
distanceFunction='cosine'
kmeanstrainrate=0.05

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
# ef=10, efConstruction=200




