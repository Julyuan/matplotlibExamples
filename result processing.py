# -*- coding: utf-8 -*-
"""
Created on Wed Aug 29 13:09:27 2018

@author: Jin Lianyuan
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scipy.interpolate import spline

#def Plot(x_coordinates, y_coordiantes, smooth = False, fmt = ''):
#    if smooth == True:
#        x_coordinates = np.array(x_coordinates)
#        y_coordiantes = np.array(y_coordiantes)
#        x_new = np.linspace(x_coordinates.min(), x_coordinates.max(), 300)
#        y_new = spline(x_coordinates, y_coordiantes, x_new, order = 2)
#        plt.plot(x_new, y_new)
#    else:
#        plt.plot(x_coordinates, y_coordiantes, fmt, label = "line1")
    
def dataSmooth(x_coordinates, y_coordiantes):
    x_coordinates = np.array(x_coordinates)
    y_coordiantes = np.array(y_coordiantes)
    x_new = np.linspace(x_coordinates.min(), x_coordinates.max(), 300)
    y_new = spline(x_coordinates, y_coordiantes, x_new, order = 2)
    return (x_new, y_new)
    
def dataTransform(x):
    y = []
    for i in list(x):
        y.append(1000.0/i)
    return y
    
def multiElementWise(x, y):
    x = list(x)
    y = list(y)
    z = []
    for i in range(len(x)):
        z.append(x[i] * y[i]/ (1024*1024))
    return z
        
def readDataFromFile(addr):
    res = pd.read_excel(addr, header=None)
    res = res.values
    return res
    
save_addr = "C:\\Users\\Jin Lianyuan\\Desktop\\test graph\\"
addr1 = r"C:\Users\Jin Lianyuan\Desktop\guomi result.xlsx"
addr2 = r"C:\Users\Jin Lianyuan\Desktop\guomi result1.xlsx"

addr3 = r"C:\Users\Jin Lianyuan\Desktop\jdk result.xlsx"
addr4 = r"C:\Users\Jin Lianyuan\Desktop\jdk result1.xlsx"

data1 = readDataFromFile(addr1) # SM4 algorithm, 第一行是数据规模，第二行是一秒执行次数(op/s)
data2 = readDataFromFile(addr2) # SM3 and SM2 algorithm, 第一行是数据规模，第二行是SM3的结果
                                # 后面两行是SM2的签名和验签
data3 = readDataFromFile(addr3) # AES algorithm, 第一行是数据规模，第二行是一秒执行次数(op/s)
data4 = readDataFromFile(addr4) # SHA256 ECDSA RSA algorithm, 第一行是数据规模，接下来各行
                                # 依次是SHA256, ECDSA签名，ECDSA验签，RSA签名，RSA验签
                                
# 对称加密结果比较
#Plot(data1[0], data1[1], data3[0], data3[1])
def symmetricResult():
    end = 5
    x1,y1 = dataSmooth(data1[0][:end], data1[1][:end])
    x2,y2 = dataSmooth(data3[0][:end], data3[1][:end])
    plt.plot(x1, y1, label = "SM4")
    plt.plot(x2, y2, label = "AES")
    plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
               ncol=3, mode='expand', borderaxespad=0.)
    plt.xlabel("data size/byte")
    plt.ylabel("speed/(op/s)")
    plt.show()
    plt.savefig(save_addr + r"symmetric1");

    x1a, y1a = dataSmooth(data1[0][:end], dataTransform(data1[1][:end]))
    x2a, y2a = dataSmooth(data3[0][:end], dataTransform(data3[1][:end]))
    plt.plot(x1a, y1a, label = "SM4")
    plt.plot(x2a, y2a, label = "AES")
    plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
               ncol=3, mode='expand', borderaxespad=0.)
    plt.xlabel("data size/byte")
    plt.ylabel("time cost/ms")
    plt.show()
    plt.savefig(save_addr + r"symmetric2");

    x1b, y1b = dataSmooth(data1[0][:end], multiElementWise(data1[0][:end], data1[1][:end]))
    x2b, y2b = dataSmooth(data3[0][:end], multiElementWise(data3[0][:end], data3[1][:end]))
    plt.plot(x1b, y1b, label = "SM4")
    plt.plot(x2b, y2b, label = "AES")
    plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
               ncol=3, mode='expand', borderaxespad=0.)
    plt.xlabel("data size/byte")
    plt.ylabel("bandwidth/(MB/s)")
    plt.show()
    plt.savefig(save_addr + r"symmetric3");
    
    begin = 6
    x1,y1 = dataSmooth(data1[0][6:], data1[1][6:])
    x2,y2 = dataSmooth(data3[0][6:], data3[1][6:])
    plt.plot(x1, y1, label = "SM4")
    plt.plot(x2, y2, label = "AES")
    plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
               ncol=3, mode='expand', borderaxespad=0.)
    plt.xlabel("data size/byte")
    plt.ylabel("speed/(op/s)")
    plt.show()
    plt.savefig(save_addr + r"symmetric4");

    x1a, y1a = dataSmooth(data1[0][6:], dataTransform(data1[1][6:]))
    x2a, y2a = dataSmooth(data3[0][6:], dataTransform(data3[1][6:]))
    plt.plot(x1a, y1a, label = "SM4")
    plt.plot(x2a, y2a, label = "AES")
    plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
               ncol=3, mode='expand', borderaxespad=0.)
    plt.xlabel("data size/byte")
    plt.ylabel("time cost/ms")
    plt.show()
    plt.savefig(save_addr + r"symmetric5");

    
    x1b, y1b = dataSmooth(data1[0][begin:], multiElementWise(data1[0][begin:], data1[1][begin:]))
    x2b, y2b = dataSmooth(data3[0][begin:], multiElementWise(data3[0][begin:], data3[1][begin:]))
    plt.plot(x1b, y1b, label = "SM4")
    plt.plot(x2b, y2b, label = "AES")
    plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
               ncol=3, mode='expand', borderaxespad=0.)
    plt.xlabel("data size/byte")
    plt.ylabel("bandwidth/(MB/s)")
    plt.show()
    plt.savefig(save_addr + r"symmetric6");

# Plot(data1[0], list(dataTransform(data1[1])), data3[0], )

# 非对称加密结果比较
def asymmetricSignResult():
    x1b, y1b = dataSmooth(data2[0], multiElementWise(data2[0], data2[2]))
    x2b, y2b = dataSmooth(data4[0], multiElementWise(data4[0], data4[2]))
    x3b, y3b = dataSmooth(data4[0], multiElementWise(data4[0], data4[4]))
    plt.plot(x1b, y1b, label = "SM2")
    plt.plot(x2b, y2b, label = "ECDSA")
    plt.plot(x3b, y3b, label = "RSA")
    plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
           ncol=3, mode='expand', borderaxespad=0.)
    plt.xlabel("data size/byte")
    plt.ylabel("bandwidth/(MB/s)")
    plt.show()
    
def asymmetricVerifyResult():
    x1b, y1b = dataSmooth(data2[0], multiElementWise(data2[0], data2[3]))
    x2b, y2b = dataSmooth(data4[0], multiElementWise(data4[0], data4[3]))
    x3b, y3b = dataSmooth(data4[0], multiElementWise(data4[0], data4[5]))
    plt.plot(x1b, y1b, label = "SM2")
    plt.plot(x2b, y2b, label = "ECDSA")
    plt.plot(x3b, y3b, label = "RSA")
    plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
       ncol=3, mode='expand', borderaxespad=0.)
    plt.xlabel("data size/byte")
    plt.ylabel("bandwidth/(MB/s)")
    plt.show()
    
# 哈希算法结果比较
def hashResult():
    x1,y1 = dataSmooth(data2[0], data2[1])
    x2,y2 = dataSmooth(data4[0], data4[1])
    plt.plot(x1,y1, label = "SM3")
    plt.plot(x2,y2, label = "SHA256")
    plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
       ncol=3, mode='expand', borderaxespad=0.)
    plt.xlabel("data size/byte")
    plt.ylabel("speed/(op/s)")
    plt.show()
    
    x1a, y1a = dataSmooth(data2[0], list(dataTransform(data2[1])))
    x2a, y2a = dataSmooth(data4[0], list(dataTransform(data4[1])))
    plt.plot(x1a,y1a, label = "SM3")
    plt.plot(x2a,y2a, label = "SHA256")
    plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
       ncol=3, mode='expand', borderaxespad=0.)
    plt.xlabel("data size/byte")
    plt.ylabel("time cost/ms")    
    plt.show()
    
    x1b, y1b = dataSmooth(data2[0], multiElementWise(data2[0], data2[1]))
    x2b, y2b = dataSmooth(data4[0], multiElementWise(data4[0], data4[1]))
    plt.plot(x1b,y1b, label = "SM3")
    plt.plot(x2b,y2b, label = "SHA256")
    plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
       ncol=3, mode='expand', borderaxespad=0.)
    plt.xlabel("data size/byte")
    plt.ylabel("bandwidth/(MB/s)")
    plt.show()

# a = [1,2,3]
# b = [4,5,6]
# c = multiElementWise(a,b)
# print(c)

# print(dataTransform(np.array([3,2,1])))
symmetricResult()
# asymmetricSignResult()
# asymmetricVerifyResult()
# hashResult()

# Plot(data2)

# print(data2)

# Plot(T, power)


#T = np.array([6,7,8,9,10,11,12])
#power = np.array([1.53E+03, 5.92E+02, 2.04E+02, 7.24E+01, 2.72E+01, 1.10E+01, 4.70E+00])

#xnew = np.linspace(T.min(), T.max(), 300)
#score = np.array([55262, 47225, 40082, 35236, 31490, 27959, 25875, 23224, 21695, 20243])
#datasize = np.array([])
#power_smooth = spline(T, power, xnew, order = 2)

#plt.plot(xnew, power_smooth)
#plt.show()