import numpy as np 


arr = np.array([[6, 3, 6, 4], [3, 1, 2, 0], [3, 1, 8, 3]]).transpose()
print('%=============原始数据(4行3列)=================')
print(arr)
arrSortedIndex = np.lexsort((arr[:, 1]))
print('%======按照x优先，y次级，z最后规则排序后=======')

print(arr[arrSortedIndex , :])

print(arrSortedIndex)

a = np.array([1,2,3,4,5])
b = np.array([50,40,30,20,10])
 
c = np.lexsort((a,b))
print(list(zip(a[c],b[c])))

print(b[c])
print(a[c])

print(a.shape,b.shape)

partion=np.array(range(8))
print(partion.shape)


d = np.zeros((8,20),int)
print(d.shape)
print(d[0],d[0].shape)

