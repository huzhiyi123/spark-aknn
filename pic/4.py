import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
def pic1(y1,y2,png):
    base=10000
    for i in range(len(y2)):
        cur = y2[i]
        t = int(base/(cur/1000))*60
        y2[i]=t
    x = np.trunc(range(1,len(y1)+1)).astype(int).tolist()
    print(x)
    fig = plt.figure(figsize=(5.5,4),dpi=300)
    # 添加Axes坐标轴实例，创建1个画板
    ax = fig.add_subplot(111)  
    # 制作第一条折现
    lin1 = ax.plot(x, y1, label='recall', color='r', marker='o')
    # 设置Y轴1
    #ax.set_ylabel('recall')
    # 使用twinx()函数实现共用一个x轴
    ax2 = ax.twinx()
    ax2.set_ylim(20000,45000)   
    # 制作第二条折现
    lin2 = ax2.plot(x, y2, label='throughput', color='green', marker='o')
    # 设置Y轴2
    #ax2.set_ylabel('throughput(/min)')
    # 合并图例
    lines = lin1+lin2
    labs = [label.get_label() for label in lines]
    ax.legend(lines,labs)
    #plt.yscale('log')
    #plt.xscale('symlog')
    # 增加网格线
    #ax.grid()
    plt.savefig(png)


x=[0.57551,0.854,0.97431,0.98359,0.98455,0.98527,0.98556,0.98632]

y=[19930.397272109985,
20740.634441375732,
20630.70845603943,
19849.08390045166,
20631.37435913086,
21262.713193893433,
22728.391885757446,
22937.6323223114]
"""
totalsearchtime 19930.397272109985 
totalsearchtime 20740.634441375732
totalsearchtime 24630.70845603943
totalsearchtime 19849.08390045166
totalsearchtime 20631.37435913086
totalsearchtime 21262.713193893433
totalsearchtime 21728.391885757446
totalsearchtime 21937.6323223114
"""
# timeUsed:  492552.9336929321 fit time 10306.30373954773


pic1(x,y,"pic2/8分区分区裁剪.png")


x=[0.52979,0.83549,0.96812, 0.98713]

y=[14582.78250694275,
15747.555255889893,
18950.77872276306,
22487.680196762085]

x=[0.57551,0.854,0.97431,0.98359,0.98455,0.98527,0.98556,0.98632]

y=[19930.397272109985,
20740.634441375732,
20630.70845603943,
19849.08390045166,
20631.37435913086,
21262.713193893433,
22728.391885757446,
22937.6323223114]

pic1(x,y,"pic2/4分区分区裁剪.png")