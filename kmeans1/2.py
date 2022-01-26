import pandas as pd
import numpy as np
path="/aknn/kmeans/siftpartition-4.csv"
path="/aknn/kmeans/mnistpartition.csv"
partitioncoldata=pd.read_csv(path,header=None,index_col=None)
print(partitioncoldata.shape)
#print(partitioncoldata.colunms)