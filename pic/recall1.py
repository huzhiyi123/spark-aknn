
# 导入相关数据包
import matplotlib.pyplot as plt
import pandas as pd

def pic(y1,y2,png):
    x = range(len(y1))

    base= 10000
    y3=[]
    for i in range(len(y2)):
        cur = y2[i]
        t = int(base/(cur/1000))*60
        y3.append(t)
    y2=y3
    fig = plt.figure(figsize=(5.5,4),dpi=100)
    # 添加Axes坐标轴实例，创建1个画板
    ax = fig.add_subplot(111)  
    # 制作第一条折现
    lin1 = ax.plot(x, y1, label='recall', color='r', marker='o')
    # 设置Y轴1
    #ax.set_ylabel('recall')
    # 使用twinx()函数实现共用一个x轴
    ax2 = ax.twinx()
    # 制作第二条折现
    lin2 = ax2.plot(x, y2, label='throughput', color='green', marker='o')
    ax2.set_ylim(20000,55000)   
    # 设置Y轴2
    #ax2.set_ylabel('throughput(/min)')
    # 合并图例
    lines = lin1+lin2
    labs = [label.get_label() for label in lines]
    ax.legend(lines,labs)
    # 增加网格线
    #ax.grid()
    plt.savefig(png)

y3=[]
y4=[]

# [15,20,50,100,150]

y1=[0.79242, 
0.83855,
0.93558,
0.98325,
0.98952]

y2=[12627.326250076294,
12581.1314582824,
15141.392946243286,
21283.632278442383,
27694.811582565308]

y3.append(y1)
y4.append(y2)

y1=[0.80132,0.84638,0.93902,0.98435,0.98919]
y2=[12839.723587036133,13250.351905822754,14602.03218460083,20491.507291793823,26843.430519104004]

y3.append(y1)
y4.append(y2)

y1=[0.80407,0.84509,0.93505,0.97458,0.97923]
y2=[13618.425846099854,12961.885452270508,14567.484140396118,18366.339206695557,22974.760055541992]

y3.append(y1)
y4.append(y2)

y1=[0.78611,0.82218,0.86352, 0.9469, 0.98383,0.98818]
y2=[13629.450559616089,13326.650857925415,14419.819831848145,15925.32753944397,22650.779247283936,29547.039031982422]

# 8分区暴力 timeUsed:  492552.9336929321 fit time 10306.30373954773


y3.append(y1)
y4.append(y2)

def func(y):
    y4=[]
    for y2 in y:
        base= 10000
        y3=[]
        for i in range(len(y2)):
            cur = y2[i]
            t = int(base/(cur/1000))*60
            y3.append(t)
        y4.append(y3)
    return y4
recalllist = y3
throughoutlist = func(y4)

res1=[]
for i in range(len(recalllist)):
    res1.append([recalllist[i],throughoutlist[i]])

part = []

for i in range(len(recalllist)):
    tmp1 = recalllist[i]
    tmp2 = throughoutlist[i]
    part.append([tmp1,tmp2])

"""

 17455.52921295166,
 0.69199,
"""
b1=[
 18866.933584213257,
 0.779,
 19116.48154258728,
 0.902,
 20182.870864868164,
 0.92358,
 21977.05125808716,
 0.96616,
 38874.08757209778,
 0.99024]

b2=[14756.173372268677,
0.84829,
15023.348569869995,
0.87351,
16748.36564064026,
0.90257,
19724.00736808777,
0.96621,
28937.756061553955,
0.99039]
 
 
b3=[14992.94400215149,
0.74817,
15987.164735794067,
0.78758,
15932.564496994019,
0.82576,
16680.29499053955,
0.88357,
22304.012298583984,
0.9592,
32686.089038848877,
0.99109]

b4=[15841.038942337036,
0.83015,
16105.677127838135,
0.86405,
16972.1896648407,
0.87093,
18847.17893600464,
0.93974,
24859.295415878296,
0.97678]

vec=[b1,b2,b3,b4]

def splitvec(vec):
    time=[]
    recall=[]
    throughout=[]
    base= 10000
    for i in range(int(len(vec)/2)):
        time.append(vec[i*2])
        recall.append(vec[i*2+1])
    for i in range(len(time)):
        cur = time[i]
        t = int(base/(cur/1000))*60
        throughout.append(t)
    return time,recall,throughout

def splitvecw(vec):
    tmp=[]
    for i in vec:
        a,b,c=splitvec(i)
        tmp.append([b,c])
    return tmp

res=splitvecw(vec)

#print(yyy)


def pic(y1,y2,png):
 
    fig = plt.figure(figsize=(4.6,4.6*0.9),dpi=500)
    # 添加Axes坐标轴实例，创建1个画板
    ax = fig.add_subplot(111)  
    # 制作第一条折现
    plt.plot(y1[0],y1[1],marker='o',label='throughput')
    plt.plot(y2[0],y2[1],marker='o',label='throughput')
    #plt.plot(y[i][0],y[i][1],marker='o',label='throughput'+str(i))

    ax.set_ylabel('throughput')
    ax.set_xlabel('recall')
    #ax.set_ylim(20000,55000)   
    plt.legend(['naive-hnsw ','kmeans-hnsw'])
    plt.yscale('log')
    plt.savefig(png)

for i in range(len(res)):
    pic(res[i],res1[i],"efpic/ef"+str(i)+".png")



