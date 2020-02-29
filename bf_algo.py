import numpy as np
from scipy.interpolate import interp1d
from math import *
from manimlib.utils.config_ops import digest_config

class MVDR(object):
    CONFIG = {
        "element_num" : 5,      #阵元个数
        "theta0" : -20, #增强方向
        "theta1" : 60,  #抑制方向
        "normal" : 0,   #是否正则化，非0值将会正则化为0到设定值内
        "snr" : 10,     #信噪比 
        "snap" : 500    #快拍数,也就是信号长度
    }
    

    def steering_vector_ula(self,theta,element_num):        #设定导向矢量
        A = np.zeros((element_num,1))
        d_lamda = 1/2
        source_num = 1
        for i in range(0,element_num-1):
            A[i] = np.exp(1j * 2 * pi * d_lamda * sin(theta*pi/180) * i)
        return A
    def randn_complex(self,n,m):
        randn_complex=(np.random.normal(loc = 0, scale = 1 ,size = (n,m)) + 1j*np.random.normal(loc = 0, scale = 1 ,size = (n,m)))/sqrt(2)
        return randn_complex
    def generate_W(self):
        s1=pow(10,self.snr/20)*np.multiply(self.steering_vector_ula(self.theta0,self.element_num),self.randn_complex(1,self.snap))
        s2=10*np.dot(self.steering_vector_ula(self.theta1,self.element_num),self.randn_complex(1,self.snap))
        n=self.randn_complex(self.element_num,self.snap)
        x=s1+s2+n
        r=np.dot(x,x.conj().T)/self.snap #自相关矩阵randn_complex
        w20=np.dot(np.linalg.inv(r),self.steering_vector_ula(self.theta0,self.element_num) ) #MVDR权值计算,忽略了常数
        return w20
    def generate_funcy(self,w20):
        p20 = np.zeros((180,1))
        y = np.zeros(180)
        theta = np.arange(-90,90)
        for i in range(0,180):
            p20[i,:] = np.dot(w20.conj().T, self.steering_vector_ula(theta[i],self.element_num))
        for i in range(0,180):
            y[i] = 10*log10(abs(p20[i,:])/np.max(np.absolute(p20)))

        #normalnization
        y_min = np.absolute(np.min(y))
        y_max = np.max(y)
        if self.normal != 0 :

            for i in range(0,180):
                y[i] = (y[i] + y_min)*(self.normal/y_max)

        #interpolate the data
        func_y = interp1d(theta,y)
        return func_y
    def __init__(self, **kwargs):
        digest_config(self, kwargs)        #使用 config 中的项定义变量
        self.w = self.generate_W()
        self.y = self.generate_funcy(self.w)

w_1 = MVDR(theta0 = 30, theta1 =70, normal = 6).w
lenth_w = np.absolute(w_1)

def normalize(w):
    W = np.absolute(w)
    scale = np.max(W)-np.min(W)
    min_value = np.min(W)
    A = (W - min_value)/scale
    return A