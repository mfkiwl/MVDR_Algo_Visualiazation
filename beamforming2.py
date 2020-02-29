from manimlib.imports import *
from math import *
import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt


def steering_vector_ula(theta,element_num):
    A = np.zeros((element_num,1))
    d_lamda = 1/2
    source_num = 1
    for i in range(0,element_num-1):
        A[i] = np.exp(1j * 2 * pi * d_lamda * sin(theta*pi/180) * i)
    return A
def randn_complex(n,m):
    randn_complex=(np.random.normal(loc = 0, scale = 1 ,size = (n,m)) + 1j*np.random.normal(loc = 0, scale = 1 ,size = (n,m)))/sqrt(2)
    return randn_complex

ele=5  #阵元个数

snr=10
    
    
snap=500    #快拍数,也就是信号长度
theta0= -20  #可以改变
theta1=60  #可以增加或减少

s1=pow(10,snr/20)*np.multiply(steering_vector_ula(theta0,ele),randn_complex(1,snap))

s2=10*np.dot(steering_vector_ula(theta1,ele),randn_complex(1,snap))

n=randn_complex(ele,snap)

x=s1+s2+n

r=np.dot(x,x.conj().T)/snap #自相关矩阵randn_complex

w20=np.dot(np.linalg.inv(r),steering_vector_ula(theta0,ele) ) #MVDR权值计算,忽略了常数

p20 = np.zeros((180,1))
y = np.zeros(180)
theta = np.arange(-90,90)
for i in range(0,180):
    p20[i,:] = np.dot(w20.conj().T, steering_vector_ula(theta[i],ele))
print(p20)
for i in range(0,180):
    y[i] = 10*log10(abs(p20[i,:])/np.max(np.absolute(p20)))
y_min = np.absolute(np.min(y))
for i in range(0,180):
    y[i] = y[i] + y_min

print(y)
func_y = interp1d(theta,y)


class lobeplot(GraphScene):
    CONFIG = {
        "x_min" : -30,
        "x_max" : 30,
        "y_min" : -30,
        "y_max" : 30,
        "x_tick_frequency": 10,
        "y_tick_frequency": 10,
        "graph_origin" : ORIGIN ,
        "function_color" : RED,
        "x_labeled_nums": range(0,90,10),

    }   
    def func_to_plot(self,x):
        return  func_y.__call__(x)

    
    def construct(self):
        self.setup_axes(animate=True)
        
        def func_to_plot(x):
            return  func_y.__call__(x)
        path = ParametricFunction(
            lambda t: np.array([
                (func_to_plot(180*t/pi)/10)*np.cos(t+pi/2),
                (func_to_plot(180*t/pi)/10)*np.sin(t+pi/2),
                0
            ]),
            color = RED,
            t_min = (-pi/2)+ 0.05,
            t_max = (pi/2)-0.05,
            step_size = 0.01
        )
        self.play(ShowCreation(path), run_time = 5)
        func_graph=self.get_graph(self.func_to_plot,color= GREEN)
        graph_lab = self.get_graph_label(func_graph, label = "angle")
        self.play(ShowCreation(func_graph), Write(graph_lab), run_time = 5)



class DemoParametricFunctions(Scene):
    def construct(self):
        path = ParametricFunction(
            lambda t: np.array([
                -np.sin(t),
                np.cos(t),
                0
            ]),
            color=RED
        )
        self.play(ShowCreation(path))
        self.wait(2)