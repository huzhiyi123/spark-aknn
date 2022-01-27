# 导入相关数据包
import matplotlib.pyplot as plt
import pandas as pd

a=[0.0923,0.7279,
0.04139999999999999,0.6626,
0.023099999999999996,0.6022,
0.021599999999999994,0.6059,
0.023999999999999994,0.5922,
0.010999999999999996,0.5659,
0.026599999999999985,0.6079,
0.018299999999999997,0.5698,
0.015099999999999988,0.5791,
0.020599999999999993,0.5884]
x=range(10)
y1=[]
y2=[]
print((len(a)/2))
for i in range(int(len(a)/2)):
    y1.append(a[i*2])
    y2.append(a[i*2+1])

"""
for i in range(len(y1)):
    y1[i] = (0.1+y1[i])/0.1
for i in range(len(y2)):
    y2[i] = (1-y2[i])
"""

print(y1)
print(y2)

x=range(10)
"""
# 5

y1=[0.0675, 0.10759999999999997, 0.046999999999999986, 0.013499999999999984, 0.014600000000000002, 0.03569999999999998, 0.028599999999999987, 0.016099999999999975, 0.019699999999999995, 0.0199] 
y2=[0.8635, 0.8156, 0.8055, 0.7697, 0.7466, 0.7642, 0.7615, 0.735, 0.7472, 0.75]


# 8
y1=[0.0592, 0.05160000000000001, 0.017499999999999988, 0.06190000000000001, 0.026100000000000012, 0.028799999999999992, 0.03520000000000001, 0.016399999999999998, 0.013800000000000007, 0.012300000000000005] 
y2=[0.7679, 0.6972, 0.6476, 0.7165, 0.6672, 0.6531, 0.6381, 0.6537, 0.6479, 0.6256]

# 10分区 
y1=[0.0923, 0.04139999999999999, 0.023099999999999996, 0.021599999999999994, 0.023999999999999994, 0.010999999999999996, 0.026599999999999985, 0.018299999999999997, 0.015099999999999988, 0.020599999999999993]
y2=[0.7279, 0.6626, 0.6022, 0.6059, 0.5922, 0.5659, 0.6079, 0.5698, 0.5791, 0.5884]

#12
y1=[0.059766666666666676, 0.03396666666666667, 0.02326666666666667, 0.024766666666666673, 0.02896666666666667, 0.031066666666666673, 0.013766666666666677, 0.02956666666666667, 0.02176666666666667, 0.012966666666666668] 
y2=[0.67, 0.592, 0.5509, 0.5705, 0.5466, 0.5497, 0.5483, 0.5619, 0.5545, 0.5267]
"""

p=10
base=1/p
for i in range(len(y1)):
    y1[i] = (base+y1[i])/base
for i in range(len(y2)):
    y2[i] = (1-y2[i])

#print(y1)
#print(y2)

def pic(x,y1,y2,png,p):
    base=1/p
    for i in range(len(y1)):
        y1[i] = (base+y1[i])/base
    for i in range(len(y2)):
        y2[i] = (1-y2[i])

    fig = plt.figure(figsize=(7,6),dpi=600)
    # 添加Axes坐标轴实例，创建1个画板
    ax = fig.add_subplot(111)  
    # 制作第一条折现
    lin1 = ax.plot(x, y1, label='cost2', color='r')
    ax.set_xlabel('repartition rate')
    # 设置Y轴1
    ax.set_ylabel('cost1')
    # 使用twinx()函数实现共用一个x轴
    ax2 = ax.twinx()
    # 制作第二条折现
    lin2 = ax2.plot(x, y2, label='cost1', color='green')
    # 设置Y轴2
    ax2.set_ylabel('cost2')
    # 合并图例
    lines = lin1+lin2
    labs = [label.get_label() for label in lines]
    ax.legend(lines,labs)
    # 增加网格线
    #ax.grid()
    plt.savefig(png)


pnglist=["5.png","8.png","10.png","12.png"]
for i in range(len(pnglist)):
    pnglist[i] = "pic/" +"1-"+ pnglist[i]


y1=[0.0675, 0.10759999999999997, 0.046999999999999986, 0.013499999999999984, 0.014600000000000002, 0.03569999999999998, 0.028599999999999987, 0.016099999999999975, 0.019699999999999995, 0.0199] 
y2=[0.8635, 0.8156, 0.8055, 0.7697, 0.7466, 0.7642, 0.7615, 0.735, 0.7472, 0.75]
pic(x,y1,y2,pnglist[0],5)
# 8
y1=[0.0592, 0.05160000000000001, 0.017499999999999988, 0.06190000000000001, 0.026100000000000012, 0.028799999999999992, 0.03520000000000001, 0.016399999999999998, 0.013800000000000007, 0.012300000000000005] 
y2=[0.7679, 0.6972, 0.6476, 0.7165, 0.6672, 0.6531, 0.6381, 0.6537, 0.6479, 0.6256]
pic(x,y1,y2,pnglist[1],8)
# 10分区 
y1=[0.0923, 0.04139999999999999, 0.023099999999999996, 0.021599999999999994, 0.023999999999999994, 0.010999999999999996, 0.026599999999999985, 0.018299999999999997, 0.015099999999999988, 0.020599999999999993]
y2=[0.7279, 0.6626, 0.6022, 0.6059, 0.5922, 0.5659, 0.6079, 0.5698, 0.5791, 0.5884]
pic(x,y1,y2,pnglist[2],10)
#12
y1=[0.059766666666666676, 0.03396666666666667, 0.02326666666666667, 0.024766666666666673, 0.02896666666666667, 0.031066666666666673, 0.013766666666666677, 0.02956666666666667, 0.02176666666666667, 0.012966666666666668] 
y2=[0.67, 0.592, 0.5509, 0.5705, 0.5466, 0.5497, 0.5483, 0.5619, 0.5545, 0.5267]
pic(x,y1,y2,pnglist[3],12)




"""
[1.9229999999999998, 1.414, 1.2309999999999999, 1.216, 1.24, 1.1099999999999999, 
1.2659999999999998, 1.183, 1.1509999999999998, 1.206]

[0.2721, 0.33740000000000003, 0.39780000000000004, 0.3941, 0.40780000000000005, 
0.43410000000000004, 0.3921, 0.4302, 0.42090000000000005, 0.41159999999999997]
"""
