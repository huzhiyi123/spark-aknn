
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


b=[17695.40500640869,
 0.62456,
 17455.52921295166,
 0.69199,
 21461.42816543579,
 0.89506,
 19116.48154258728,
 0.902,
 20182.870864868164,
 0.92358,
 21977.05125808716,
 0.96616,
 38874.08757209778,
 0.99024]


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

a,b,c = splitvec(b)
print(a,b,c)


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

yyy = func(y4)
#print(yyy)


def pic(y1,y2,png):
    fig = plt.figure(figsize=(9,4),dpi=700)
    # 添加Axes坐标轴实例，创建1个画板
    ax = fig.add_subplot(111)  
    # 制作第一条折现
    l = []
    for i in range(len(y2)):
        plt.plot(y1,y2[i],marker='o',label='throughput'+str(i))
    # 增加网格线
    #ax.grid()
    ax.set_ylabel('time/ms')
    ax.set_xlabel('ef_constructuion')
    #ax.set_ylim(20000,55000)   
    plt.legend(['4 partions','6 partitions' ])
    plt.yscale('log')
    plt.savefig(png)





def pic2(y1,y2,png):
    x = range(len(y1))
    fig = plt.figure(figsize=(5.5,4.2),dpi=100)
    # 添加Axes坐标轴实例，创建1个画板
    ax = fig.add_subplot(111)  
    # 制作第一条折现
    l = []
    for i in range(len(y1)):
        plt.plot(y1[i],y2[i])
    # 增加网格线
    #ax.grid()
    plt.savefig(png)




def pic1(y1,y2,png):
    x = range(len(y1))
    fig = plt.figure(figsize=(5.5,4.2),dpi=100)
    # 添加Axes坐标轴实例，创建1个画板
    ax = fig.add_subplot(111)  
    # 制作第一条折现
    lin1 = ax.plot(x, y1, label='recall', color='r')
    # 设置Y轴1
    #ax.set_ylabel('recall')
    # 使用twinx()函数实现共用一个x轴
    ax2 = ax.twinx()
    # 制作第二条折现
    lin2 = ax2.plot(x, y2, label='throughput', color='green')
    # 设置Y轴2
    #ax2.set_ylabel('throughput(/min)')
    # 合并图例
    lines = lin1+lin2
    labs = [label.get_label() for label in lines]
    ax.legend(lines,labs)
    # 增加网格线
    #ax.grid()
    plt.savefig(png)


"""
pic2(y3,yyy,"test.png")

for i in range(4):
    print(len(y3[i]),len(y4[i]))
    pic(y3[i],y4[i],"pic1/" +"ef"+str(i)+".png")
"""