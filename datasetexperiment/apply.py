import pandas as pd
import numpy as np
l=10
n=3
df=pd.DataFrame(columns=['id','partition'])#(np.arange(l),columns=['id'])
df['id']=np.array(range(l))
partition=np.array(range(l*n)).reshape(l,n)
df['partition']=partition.tolist()
print(df)

def func(x):
    x[0]+=pmap[x[0]]
    x[1]+=pmap[x[1]]
    x[2]+=pmap[x[2]]
    return x
pmap=range(100,1000)

def getrepartitionmap(repartitionres,k1,k2):
    res = np.zeros(k2,int)
    for i in range(k1):
        cur = repartitionres[i]
        for j in cur:
            res[j]=i
    return res

res=range(100,1000)

df['partition'].apply(lambda x:func(x))

print(df)