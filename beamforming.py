from manimlib.imports import *
from math import *
from scipy.interpolate import interp1d
from manimlib.utils.config_ops import digest_config
from manimlib.once_useful_constructs.light import *

class MVDR(object):
    CONFIG = {
        "element_num" : 9,      #阵元个数
        "theta0" : -20, #增强方向
        "theta1" : 60,  #抑制方向
        "normal" : 6,   #是否正则化，非0值将会正则化为0到设定值内
        "snr" : 10,     #信噪比 
        "snap" : 500    #快拍数,也就是信号长度
    }
    def __init__(self, **kwargs):
        digest_config(self, kwargs)        #使用 config 中的项定义变量
        self.w = self.generate_W()
        self.y = self.generate_funcy(self.w)

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
        y_min = np.min(y)
        y_range = np.absolute(np.max(y)-np.min(y))
        if self.normal != 0 :
            y = self.normal*(y - y_min)/y_range
        #interpolate the data
        #print("w:\n")
        print(w20)
        func_y = interp1d(theta,y)
        return func_y
    def set_theta0(self,theta):
        self.theta0 = theta
        #print(theta)
        self.w = self.generate_W()
        #print(self.w)
        self.y = self.generate_funcy(self.w)
    def set_theta1(self,theta):
        self.theta1 = theta
        self.w = self.generate_W()
        self.y = self.generate_funcy(self.w)
def normalize1(w):
    W = np.absolute(w)
    scale = np.max(W)-np.min(W)
    min_value = np.min(W)
    A = (W - min_value)/scale
    return A

class Begining_title(Scene):
    def construct(self):
        line_1 = TextMobject("Introduction of ","Beamforming").scale(1.5).set_color_by_tex("Beamforming", color = BLUE)
        line_2 = TextMobject("Beamforming").set_color_by_tex("Beamforming", color = BLUE)
        line_3 = TextMobject("What is ","Beamforming").set_color_by_tex("Beamforming", color = BLUE)
        line_4 = TextMobject("What does","Beamforming","mean?").set_color_by_tex("Beamforming", color = BLUE)
        self.play(Write(line_1))
        self.wait()
        self.play(Transform(line_1,line_2))
        self.wait()
        self.clear()
        self.play(
            line_2.shift, 1*UP,
            line_2.scale, 2
        )

        svg_5g = SVGMobject("/Users/neoncloud/Documents/GitHub/manim-0.1.10/study/5G.svg").scale(0.5).move_to([-3,-1,0])
        svg_5g[1].set_color(BLUE)
        svg_5g[0].set_color(RED)

        svg_wifi = SVGMobject("/Users/neoncloud/Documents/GitHub/manim-0.1.10/study/WiFi_Logo.svg").scale(0.5).move_to([0,-1,0])
        svg_wifi[1].set_color(BLACK)
        svg_wifi[2].set_color(BLACK)
        svg_wifi[3].set_color(BLACK)

        svg_3dot = SVGMobject("/Users/neoncloud/Documents/GitHub/manim-0.1.10/study/3dot_1.svg").scale(0.1).move_to([3,-1,0])


        line = []
        line.append(Line(start = line_2.get_bottom(), end = [-3,0,0]))
        line.append(Line(start = line_2.get_bottom(), end = [0,0,0]))
        line.append(Line(start = line_2.get_bottom(), end = [3,0,0]))


        self.wait()
        self.play(
            Write(svg_5g),
            GrowArrow(line[0]),
            run_time = 0.5
        )
        self.wait()
        self.play(
            Write(svg_wifi),
            GrowArrow(line[1]),
            run_time = 0.5
        )
        self.wait()
        self.play(
            Write(svg_3dot),
            GrowArrow(line[2]),
            run_time = 0.5
        )
        self.wait()
        self.play(
            *[FadeOut(i) for i in [*line, svg_5g, svg_3dot, svg_wifi]],
            Transform(line_2,line_3)
        )
        
        self.wait()
        self.clear()
        self.play(
            Transform(line_3,line_4)
        )

        self.wait()
        self.clear()
        self.play(
            Transform(line_4,line_1.move_to(ORIGIN))
        )
        self.wait()

        new_tech_bg_array = [
            "3-D Printing",
            "Artificial Intelligence",
            "Generative Adversarial Networks",
            "Virtual Reality",
            "Beamforming",
            "Block Chain",
            "Cloud Services",
            "IoT",
            "Autonomous Driving",
            "Argmented Reality",
            "Quantum Computing"
        ]
        new_tech1 = TextMobject(*new_tech_bg_array[0:3])
        new_tech2 = TextMobject(*new_tech_bg_array[3:7])
        new_tech3 = TextMobject(*new_tech_bg_array[7:11])

        for i in [*new_tech1,*new_tech2,*new_tech3]:
            i.set_color(GRAY)
            i.scale(np.random.random_sample()+0.15)
        new_tech = VGroup(new_tech1, new_tech2, new_tech3).arrange(
            UP,
            aligned_edge = LEFT,
            buff=0.4
        )
        new_tech[1][1].set_color(WHITE).scale(2.5/new_tech[1][1].get_width())
        self.clear()
        self.play(
            ReplacementTransform(line_1, new_tech[1][1]),
            *[GrowFromCenter(i) for i in new_tech]
        )
        self.wait()
        self.play(
            FadeOut(new_tech),
            line_1.move_to, UP,
            line_1.scale, 2,
            line_1.set_color, BLUE
        )

        svg_processor = SVGMobject("/Users/neoncloud/Documents/GitHub/manim-0.1.10/study/cpu.svg").scale(0.5).move_to([3,-1,0])
        svg_station = SVGMobject("/Users/neoncloud/Documents/GitHub/manim-0.1.10/study/station.svg").scale(0.5).move_to([-3,-1,0])
        plus = TexMobject("+").scale(2).move_to([0,-1,0])

        self.play(
            Write(svg_station),
            run_time = 0.5
        )
        self.wait()
        self.play(
            Write(svg_processor),
            Write(plus),
            run_time = 0.5
        )
        self.wait()
        self.play(
            FadeOutAndShiftDown(svg_processor),
            FadeOutAndShiftDown(svg_station),
            FadeOutAndShiftDown(plus),
            line_1.move_to, ORIGIN
        )
        self.wait()
        wave = []
        for i in range(0,5):
            wave.append(Arc(angle = 2*pi ,radius = 0.01, fill_color = YELLOW, fill_opacity = 0.1, color = YELLOW, opacity = 0.1))

        wave_target = []
        for i in range(0,5):
            wave_target.append(Arc(angle = 2*pi, radius = 6, fill_color = YELLOW, fill_opacity = 0, color = None, opacity = 0))

        self.add(*wave,line_1)
        self.play(
            AnimationGroup(
                *[
                    Transform(i.copy(),j.copy()) for i,j in zip(wave,wave_target)
                ],
                lag_ratio = 0.1
            ),
            run_time = 3
        )
        self.wait()
        self.play(
            AnimationGroup(
                *[
                    Transform(i.copy(),j.copy()) for i,j in zip(wave,wave_target)
                ],
                lag_ratio = 0.1
            ),
            run_time = 3
        )
        for i in wave:
            i.__init__(start_angle = pi/3, angle = pi/6, radius = 0.01, fill_color = YELLOW, fill_opacity = 0.1, color = YELLOW, opacity = 0.1)
        for i in wave_target:
            i.__init__(start_angle = pi/3, angle = pi/6, radius = 6, fill_color = YELLOW, fill_opacity = 0, color = None, opacity = 0)

        self.play(
            AnimationGroup(
                *[
                    Transform(i.copy(),j.copy()) for i,j in zip(wave,wave_target)
                ],
                lag_ratio = 0.1
            ),
            run_time = 3
        )
        self.wait()
        self.play(
            AnimationGroup(
                *[
                    Transform(i.copy(),j.copy()) for i,j in zip(wave,wave_target)
                ],
                lag_ratio = 0.1
            ),
            run_time = 3
        )
        mvdr_text1 = TextMobject(
            "M",
            "V",
            "D",
            "R"
        ).set_color_by_gradient(GREEN,BLUE).scale(1.5)
        mvdr_text2 = TextMobject(
            "Minimum",
            "Variance",
            "Distortionless",
            "Response"
        ).set_color_by_gradient(GREEN,BLUE).scale(0.8)
        self.wait()
        self.play(
            line_1.shift, UP,
            Write(mvdr_text1)
        )
        self.wait()
        self.play(
            Transform(mvdr_text1,mvdr_text2)
        )



class Rotating_vector(Scene):
    def construct(self):
        angle_of_vec_1 = ValueTracker(0)
        angle_of_vec_2 = ValueTracker(0)
        vector_1 = Line(start = ORIGIN, end = [2,0,0])
        vector_2 = Line(start = ORIGIN, end = [1,0,0])

        def updater_vec_1(obj):
            obj.set_angle(
                angle_of_vec_1.get_value()
            )
        
        def updater_vec_2(obj):
            obj.set_angle(
                angle_of_vec_2.get_value()
            )
            obj.put_start_and_end_on(
                vector_1.get_end(),
                vector_1.get_end() + vector_2.get_vector()  #move vec_2 to the end of vec_1 
            )


        vector_1.rotate(angle_of_vec_1.get_value(), about_point = vector_1.get_start())
        vector_2.rotate(angle_of_vec_1.get_value(), about_point = vector_2.get_start())


        self.add(vector_1, vector_2)

        self.play(
            angle_of_vec_1.increment_value,2*PI,
            UpdateFromFunc(vector_1,updater_vec_1),
            angle_of_vec_2.increment_value,5*PI,
            UpdateFromFunc(vector_2,updater_vec_2),
            rate_func = linear,
            run_time = 10
        )
class Slider_mechanism(Scene):
    def construct(self):
        text_1 = TextMobject("A demo of updater function")
        text_1_trans = TextMobject("A demo of updater function").scale(0.5).to_corner(UP+LEFT)

        angle_of_vec_1 = ValueTracker(0)
        vector_1 = Line(start = ORIGIN, end = [1,0,0])
        vector_2 = Line(start = vector_1.get_end(), end = [4, 0, 0], color = GREEN)
        vector_3 = Line(color = BLUE).add_updater(lambda m: m.put_start_and_end_on(vector_1.get_start(),vector_2.get_end()))

        lable_x = TextMobject("x", color = GREEN).add_updater(lambda m: m.move_to(vector_2))
        lable_r = TextMobject("R").add_updater(lambda m: m.move_to(vector_1))
        lable_l = TextMobject("L").add_updater(lambda m: m.move_to(vector_3))

        number_line = NumberLine()
        circle = DashedVMobject(Circle(radius = 1, color = YELLOW))
        box = Rectangle(height = 0.5, width = 0.8, color = BLUE, fill_color = BLUE, fill_opacity = 0.75).add_updater(lambda m: m.move_to(vector_2.get_end()))

        def updater_vec_1(obj):
            obj.set_angle(
                angle_of_vec_1.get_value()
            )
        
        def updater_vec_2(obj):
            obj.put_start_and_end_on(
                vector_1.get_end(),
                [sqrt(9 - 1 + pow( cos(angle_of_vec_1.get_value()), 2 ) ) + cos(angle_of_vec_1.get_value()), 0, 0]
            )
        

        self.play(Write(text_1))
        self.wait()
        self.play(Transform(text_1, text_1_trans))
        self.wait()

        objs = [box, number_line, circle, vector_1, vector_2, vector_3, lable_x, lable_r, lable_l] 
        self.play(*[
            FadeIn(i) for i in objs
        ]   #dynamic args
        )
        

        self.add(
            *[
                j for j in objs
            ]
        )
        self.play(
            angle_of_vec_1.increment_value,8*PI,
            UpdateFromFunc(vector_1,updater_vec_1),
            UpdateFromFunc(vector_2,updater_vec_2),
            rate_func = linear,
            run_time = 10
        )
class Arrays(Scene):
    def construct(self):
        W_1 = Circle(radius = 0.5, arc_center = [-4,0,0], color = YELLOW)
        W_2 = Circle(radius = 0.5, arc_center = [-2,0,0], color = YELLOW)
        W_3 = Circle(radius = 0.5, arc_center = [0,0,0], color = YELLOW)
        W_4 = Circle(radius = 0.5, arc_center = [2,0,0], color = YELLOW)
        W_5 = Circle(radius = 0.5, arc_center = [4,0,0], color = YELLOW)

        W_1_label = TextMobject("W${^*_1}$").move_to(W_1.get_center())
        W_2_label = TextMobject("W${^*_2}$").move_to(W_2.get_center())
        W_3_label = TextMobject("W${^*_3}$").move_to(W_3.get_center())
        W_4_label = TextMobject("W${^*_4}$").move_to(W_4.get_center())
        W_5_label = TextMobject("W${^*_5}$").move_to(W_5.get_center())

        Sigma_circel = Circle(radius = 0.6, arc_center = [0,-3,0], color = BLUE, fill_color = WHITE, fill_opacity = 1)
        Sigma = TexMobject("\\Sigma", color = BLACK).move_to(Sigma_circel.get_center())
        

        line_1 = Line(start = W_1.get_center()+[0,2,0], end = W_1.get_top())
        line_2 = Line(start = W_2.get_center()+[0,2,0], end = W_2.get_top())
        line_3 = Line(start = W_3.get_center()+[0,2,0], end = W_3.get_top())
        line_4 = Line(start = W_4.get_center()+[0,2,0], end = W_4.get_top())
        line_5 = Line(start = W_5.get_center()+[0,2,0], end = W_5.get_top())

        arrow_1 = Arrow(start = W_1.get_bottom(), end = Sigma_circel.get_top(),tip_length=0.2)
        arrow_2 = Arrow(start = W_2.get_bottom(), end = Sigma_circel.get_top(),tip_length=0.2)
        arrow_3 = Arrow(start = W_3.get_bottom(), end = Sigma_circel.get_top(),tip_length=0.2)
        arrow_4 = Arrow(start = W_4.get_bottom(), end = Sigma_circel.get_top(),tip_length=0.2)
        arrow_5 = Arrow(start = W_5.get_bottom(), end = Sigma_circel.get_top(),tip_length=0.2)

        group1 = VGroup(W_1, W_1_label, W_2, W_2_label, W_3, W_3_label, W_4, W_4_label, W_5, W_5_label, Sigma_circel, Sigma, line_1, line_2, line_3, line_4, line_5, arrow_1, arrow_2, arrow_3, arrow_4, arrow_5)
        group2 = copy.deepcopy(group1)          #using deepcopy to keep group1 untouched
        group2.scale_in_place(0.5).shift(DOWN)
        
        obj_W = [W_1, W_2, W_3, W_4, W_5]
        obj_W_lable = [W_1_label, W_2_label, W_3_label, W_4_label, W_5_label]
        obj_sigma = [Sigma_circel, Sigma]
        obj_line = [line_1, line_2, line_3, line_4, line_5]
        obj_arrow = [arrow_1, arrow_2, arrow_3, arrow_4, arrow_5]
        #[W_1, W_1_label, W_2, W_2_label, W_3, W_3_label, W_4, W_4_label, W_5, W_5_label, Sigma_circel, Sigma, line_1, line_2, line_3, line_4, line_5, arrow_1, arrow_2, arrow_3, arrow_4, arrow_5]
        #self.add(group1,group2)
        self.play(
            *[
                ShowCreation(i) for i in obj_W+obj_W_lable+obj_line+obj_arrow+obj_sigma
            ],run_time = 2
        )
        self.wait()

        self.play(
            Transform(group1,group2)
        )
class Arrays2(MovingCameraScene):

    mvdr_1 = MVDR(theta0 = 20, theta1 =55, normal = 4)
    mvdr_2 = MVDR(theta0 = 45, theta1 =15, normal = 4)
    mvdr_3 = MVDR(theta0 = 60, theta1 =15, normal = 4)
    mvdr_4 = MVDR(theta0 = 60, theta1 =30, normal = 4)
    w_1 = normalize1(mvdr_1.w)
    w_2 = normalize1(mvdr_2.w)
    w_3 = normalize1(mvdr_3.w)
    w_4 = normalize1(mvdr_4.w)

    test_plot1_array = np.zeros(180)+3
    test_plot1_array[60] = 4
    test_plot1_func = interp1d(range(-90,90),test_plot1_array)

    test_plot2_array = np.zeros(180)+3
    test_plot2_array[60] = 4
    test_plot2_array[30] = 0
    test_plot2_func = interp1d(range(-90,90),test_plot2_array)


    angle_of_theta0 = ValueTracker(20*pi/180)
    angle_of_theta1 = ValueTracker(60*pi/180)

    def Updater_of_plot2(self,plot):
            self.mvdr_2.set_theta0(self.angle_of_theta0.get_value())
            self.mvdr_2.set_theta1(self.angle_of_theta1.get_value())
            plot.__init__(function = lambda t: np.array([
                (self.mvdr_2.y.__call__(degrees(t)))*np.cos(t+pi/2),
                (self.mvdr_2.y.__call__(degrees(t)))*np.sin(t+pi/2),
                0
            ]))
            plot.shift([0,-1,0])

    def construct(self):
        theta_angle = ValueTracker(90)
        num_line = NumberLine(include_ticks = False)
        Sigma_circel = Circle(radius = 0.4, arc_center = [0,-3,0], color = BLUE, fill_color = WHITE, fill_opacity = 1)
        Sigma = TexMobject("\\Sigma", color = BLACK).move_to(Sigma_circel.get_center())
        #array_circle = DashedVMobject(Arc(start_angle = 0, angle = pi ,color = PURPLE, radius = 4).shift([0,-1,0]))
        #vector_to_theta0 = Arrow(start = [0,-1,0], end = [0,3,0],stroke_width = 1.5,tip_length=0.1)
        text_1_theta0 = TexMobject("\\theta_{0} = 20").move_to([-6,1,0]).scale(0.5)
        text_1_theta1 = TexMobject("\\theta_{1} = 55").next_to(text_1_theta0,UP).scale(0.5)

        text_2_theta0 = TexMobject("\\theta_{0} = 45").move_to([-6,1,0]).scale(0.5)
        text_2_theta1 = TexMobject("\\theta_{1} = 15").next_to(text_2_theta0,UP).scale(0.5)

        text_3_theta0 = TexMobject("\\theta_{0} = 60").move_to([-6,1,0]).scale(0.5)
        text_3_theta1 = TexMobject("\\theta_{1} = 15").next_to(text_3_theta0,UP).scale(0.5)

        text_4_theta0 = TexMobject("\\theta_{0} = 60").move_to([-6,1,0]).scale(0.5)
        text_4_theta1 = TexMobject("\\theta_{1} = 30").next_to(text_4_theta0,UP).scale(0.5)

        theta0_indecator = DashedLine(start = [0,0.5,0], end = [0,-6,0], color = GREEN) 
        theta1_indecator = DashedLine(start = [0,0.5,0], end = [0,-6,0], color = GREEN) 

        text_formula_array = [
            "y(t)",
            "=",
            "\\sum_{i=1}^{8}",
            "w_{i}^{*}",
            "x_i(t)",
            "=",
            "\\mathbf{w}",
            "^{H}",
            "\\mathbf{x}(t)"
        ]
        text_formula = TexMobject(*text_formula_array).move_to(1.5*UP)

        obj_sigma = [Sigma_circel, Sigma]
        dot = []
        dot_lable = []
        W = []
        W_lable = []
        line = []
        arrow = []

        for i in range(0,9):
            dot.append(Dot(point=[-6+1.5*i, 0, 0], color = YELLOW))
            dot_lable.append(TexMobject("x_%d"%i).next_to(dot[i],UP).scale(0.7))
            W.append(Circle(radius = 0.3, arc_center = dot[i].get_center()+[0,-0.75,0], color = YELLOW, fill_color = YELLOW))
            W_lable.append(TextMobject("W${^*_%d}$"%i).move_to(W[i].get_center()).scale(0.5))
            line.append(Arrow(start = dot[i].get_center(), end = W[i].get_top(), stroke_width = 1.5))
            arrow.append(Arrow(start = W[i].get_bottom(), end = Sigma_circel.get_top(),tip_length=0.1, buff = SMALL_BUFF, stroke_width = 1.5))

        plot_1 = ParametricFunction(
            lambda t: np.array([
                3.87*t,
                (self.mvdr_1.y.__call__(degrees(t))),
                0
            ]),
            color = RED,
            t_min = -pi/2 + 0.02,
            t_max = pi/2 - 0.02,
            step_size = 0.01
        ).shift([0,-3.5,0])

        plot_2 = ParametricFunction(
            lambda t: np.array([
                3.87*t,
                (self.mvdr_2.y.__call__(degrees(t))),
                0
            ]),
            color = RED,
            t_min = -pi/2 + 0.02,
            t_max = pi/2 - 0.02,
            step_size = 0.01
        ).shift([0,-3.5,0])

        plot_3 = ParametricFunction(
            lambda t: np.array([
                3.87*t,
                (self.mvdr_3.y.__call__(degrees(t))),
                0
            ]),
            color = RED,
            t_min = -pi/2 + 0.02,
            t_max = pi/2 - 0.02,
            step_size = 0.01
        ).shift([0,-3.5,0])  

        plot_4 = ParametricFunction(
            lambda t: np.array([
                3.87*t,
                (self.mvdr_4.y.__call__(degrees(t))),
                0
            ]),
            color = RED,
            t_min = -pi/2 + 0.02,
            t_max = pi/2 - 0.02,
            step_size = 0.01
        ).shift([0,-3.5,0])  

        plot_test0 = ParametricFunction(
            lambda t: np.array([
                3.87*t,
                4,
                0
            ]),
            color = RED,
            t_min = -pi/2 + 0.02,
            t_max = pi/2 - 0.02,
            step_size = 0.01
        ).shift([0,-3.5,0])  
        
        plot_test1 = ParametricFunction(
            lambda t: np.array([
                3.87*t,
                (self.test_plot1_func.__call__(degrees(t))),
                0
            ]),
            color = RED,
            t_min = -pi/2 + 0.02,
            t_max = pi/2 - 0.02,
            step_size = 0.01
        ).shift([0,-3.5,0])   

        plot_test2 = ParametricFunction(
            lambda t: np.array([
                3.87*t,
                (self.test_plot2_func.__call__(degrees(t))),
                0
            ]),
            color = RED,
            t_min = -pi/2 + 0.02,
            t_max = pi/2 - 0.02,
            step_size = 0.01
        ).shift([0,-3.5,0])  

        plot_grid = NumberPlane(center_point = [0,0.5,0],number_line_config = {"include_tip": True}, y_axis_config = {"include_ticks": False},)
        #plot_grid.x_axis.numbers_with_elongated_ticks = range(-90,90,15)
        plot_grid.y_axis.shift(4*DOWN)
        plot_grid.background_lines.shift(4*DOWN)
        plot_grid.faded_lines.shift(4*DOWN)
        lable_theta = []
        for i in range (0,13):
            lable_theta.append(TexMobject("%d^\\circ"%((i-6)*15)).move_to([i-6,0.7,0]).scale(0.4))
        #label_x = TexMobject("\\theta").next_to(plot_grid.x_axis).scale(0.5)
        label_y = TexMobject("0 dB").move_to([6.5,0,0]).scale(0.5)
        #lable_theta = range(-90,90,15)
        self.play(
            *[
                ShowCreation(i) for i in [num_line] + dot
            ],
            run_time = 2
        )
        self.wait()
        self.play(
            *[Write(i) for i in  dot_lable]
        )
        self.wait()
        """ self.play(
            AnimationGroup(
                *[
                    Flash(i) for i in dot
                ],
                lag_ratio = 0.1
            ),
            run_time = 1
        ) """
        wave = []
        for i in range(0,3):
            wave.append(Arc(angle = 2*pi ,radius = 0.01, fill_color = YELLOW, fill_opacity = 0.1, color = YELLOW, opacity = 0.1).move_to([0,10,0]))

        wave_target = []
        for i in range(0,3):
            wave_target.append(Arc(angle = 2*pi, radius = 15, fill_color = YELLOW, fill_opacity = 0, color = None, opacity = 0)
            .move_to([0,10,0]))

        self.play(
            AnimationGroup(
                *[
                    Transform(i.copy(),j.copy()) for i,j in zip(wave,wave_target)
                ],
                lag_ratio = 0.4
            ),
            run_time = 4
        )
        self.play(
            *[
                ShowCreation(i) for i in line
            ],
            *[
                ShowCreation(i) for i in W + W_lable
            ]
        )
        self.wait()
        self.play(
            *[
                ShowCreation(i) for i in arrow + obj_sigma
            ]
        )
        self.wait()
        self.play(
            *[ReplacementTransform(i.copy(),text_formula[6]) for i in W_lable],
            *[ReplacementTransform(i.copy(),text_formula[8]) for i in dot_lable],
            Write(text_formula)
        )
        group = Group(*(dot + dot_lable + W + W_lable + line))
        def Updater_of_arrow_tip(obj):
            index = arrow.index(obj)
            obj.put_start_and_end_on(start = W[index].get_bottom(), end = Sigma_circel.get_top())
        #self.play(ShowCreation(array_circle),ShowCreation(vector_to_theta0))
        self.wait()
        self.play(
            group.shift, 3.5*UP,
            group.scale, 0.7,
            num_line.shift, 3.5*UP,
            Sigma.shift, 4.5*UP,
            Sigma.scale, 0.8,
            Sigma_circel.shift, 4.5*UP,
            Sigma_circel.scale, 0.8,
            text_formula.scale, 0.5,
            text_formula.move_to,[2.5,1.5,0],  
            *[UpdateFromFunc(i,Updater_of_arrow_tip) for i in arrow]
        )
        self.play(
            #theta_angle.set_value,70+90,
            #UpdateFromFunc(vector_to_theta0,Updater_of_theta_vector),
            ShowCreation(plot_test0),
            ShowCreation(plot_grid),
            #ShowCreation(label_x),
            ShowCreation(label_y),
            *[Write(i) for i in lable_theta],
            #*[Transform(i,deepcopy(i).set_style(fill_opacity = j)) for i,j in zip(W,self.w_1)],
            
        )
        
        self.play(
            ReplacementTransform(plot_test0,plot_test1),
            theta0_indecator.move_to, [0,-2.75,0]+2*LEFT
        )
        self.wait()
        self.play(
            ReplacementTransform(plot_test1,plot_test2),
            theta1_indecator.move_to, [0,-2.75,0]+4*LEFT
        )
        self.wait()
        self.play(
            ReplacementTransform(plot_test2,plot_1),
            *[Transform(i,deepcopy(i).set_style(fill_opacity = j)) for i,j in zip(W,self.w_1)],
            Write(text_1_theta0),
            Write(text_1_theta1),
            theta0_indecator.move_to, [0,-2.75,0]+RIGHT*20/15,
            theta1_indecator.move_to, [0,-2.75,0]+RIGHT*55/15,
        )
        self.wait()
        self.play(
            #theta_angle.set_value,10+90,
            #UpdateFromFunc(vector_to_theta0,Updater_of_theta_vector),
            ReplacementTransform(plot_1,plot_2),
            ReplacementTransform(text_1_theta0,text_2_theta0),
            ReplacementTransform(text_1_theta1,text_2_theta1),
            *[Transform(i,deepcopy(i).set_style(fill_opacity = j)) for i,j in zip(W,self.w_2)],
            theta0_indecator.move_to, [0,-2.75,0]+RIGHT*15/15,
            theta1_indecator.move_to, [0,-2.75,0]+RIGHT*45/15,
        )
        self.wait()
        self.play(
            ReplacementTransform(plot_2,plot_3),
            ReplacementTransform(text_2_theta0,text_3_theta0),
            ReplacementTransform(text_2_theta1,text_3_theta1),            
            *[Transform(i,deepcopy(i).set_style(fill_opacity = j)) for i,j in zip(W,self.w_3)],
            theta0_indecator.move_to, [0,-2.75,0]+RIGHT*15/15,
            theta1_indecator.move_to, [0,-2.75,0]+RIGHT*60/15,
        )
        self.wait()
        self.play(
            ReplacementTransform(plot_3,plot_4),
            ReplacementTransform(text_3_theta0,text_4_theta0),
            ReplacementTransform(text_3_theta1,text_4_theta1),
            *[Transform(i,deepcopy(i).set_style(fill_opacity = j)) for i,j in zip(W,self.w_4)],
            theta0_indecator.move_to, [0,-2.75,0]+RIGHT*60/15,
            theta1_indecator.move_to, [0,-2.75,0]+RIGHT*30/15,
        )
class Algo_proof_optimization(Scene):
    def construct(self):
        #optimiazation definition
        text_1 = [
            "Minimize",
            "_{\\mathbf{w}}",
            "\\hspace{0.1cm}\\mathbf{E}(y^{2}(t))",
            "=",
            "\\mathbf{w}",
            "^{H}",
            "\\mathbf{R}",
            "\\mathbf{w}"
        ]
        mob_1 = TexMobject(*text_1).shift(0.5*UP)
        mob_1[1].set_color(GREEN)
        mob_1[4].set_color(GREEN)
        mob_1[5].set_color(BLUE)
        mob_1[6].set_color(RED)
        mob_1[7].set_color(GREEN)
        mob_1_copy = deepcopy(mob_1)
        brace_1 = Brace(mob_1[6],UP)
        brace_1_text = brace_1.get_tex("\\mathbf{E}(\\mathbf{x}(t)\\mathbf{x}^{H}(t))")
        brace_2_text = brace_1.get_text("Autocorrelation Matrix")


        text_2 = [
            "Subject\\hspace{0.1cm} to \\hspace{0.2cm}",
            "\\mathbf{w}",
            "^{H}",
            "\\mathbf{a}(\\theta_{0})",
            "=",
            "1"
        ]
        mob_2 = TexMobject(*text_2)
        mob_2.next_to(mob_1,DOWN)
        mob_2[1].set_color(GREEN)
        mob_2[2].set_color(BLUE)
        mob_2_copy = deepcopy(mob_2)
        brace_2 = Brace(mob_2[3],DOWN)
        brace_3_text_array = [
            "\\mathbf{a}(\\theta)=[1, e^{(-j\\frac{2\\pi}{",
            "\\lambda_x}",
            "dsin(\\theta))}",
            ",",
            "\\cdots,e^{(-j\\frac{2\\pi}{",
            "\\lambda_x}",
            "(N-1)d sin(\\theta))}]^{\\rm T}"
        ]
        brace_3_text = brace_2.get_tex(*brace_3_text_array).scale(0.8)
        framebox1 = SurroundingRectangle(brace_3_text[1])
        framebox2 = SurroundingRectangle(brace_3_text[5])
        
        #lambda optim

        formula_1 = [
            "\\mathcal{L}",
            "(",
            "\\mathbf{w},",
            "\\lambda)",
            "= \\frac{1}{2}",
            "\\mathbf{w}",
            "^{H}",
            "\\mathbf{R}",
            "\\mathbf{w}",
            " +\\lambda(",
            "\\mathbf{w}",
            "^{H}",
            "\\mathbf{a}(\\theta_0)",
            " -1)"
        ]
        mob_3 = TexMobject(*formula_1)

        corres_ele1_in_formula = [
            mob_1[1],
            #mob_1[3:7],
            mob_1[4],
            mob_1[5],
            mob_1[6],
            mob_1[7],
            #mob_2[0:4],
            mob_2[1],
            mob_2[2],
            mob_2[3],
        ]
        corres_ele2_in_formula = [
            mob_3[2].set_color(GREEN),
            #formula_1[3:7],
            mob_3[5].set_color(GREEN),
            mob_3[6].set_color(BLUE),
            mob_3[7].set_color(RED),
            mob_3[8].set_color(GREEN),
            #formula_1[8:12]
            mob_3[10].set_color(GREEN),
            mob_3[11].set_color(BLUE),
            mob_3[12],
        ]
        mob_3_copy = copy.deepcopy(mob_3)
        rest_ele_in_formula = copy.copy(mob_3).remove(*corres_ele2_in_formula)

        formula_2 = [
            "\\partial",
            "\\mathcal{L}",
            "(",
            "\\mathbf{w}",
            ",",
            "\\lambda)}",
            "\\over",
            "\\partial",
            "\\mathbf{w}}",
            " = ",
            "\\mathbf{R}",
            "\\mathbf{w}",
            "+",
            "\\lambda",
            "\\mathbf{a}(\\theta_0)",
            "=",
            "\\mathbf{0}"
        ]
        mob_4 = TexMobject(*formula_2)
        mob_4.set_color_by_tex("\\mathbf{w}",GREEN)
        mob_4.set_color_by_tex("\\mathbf{R}",RED)

        # expression of W
        formula_2_partital = [
            "\\mathbf{R}",
            "\\mathbf{w}",
            "+",
            "\\lambda",
            "\\mathbf{a}(\\theta_0)",
            "=",
            "\\mathbf{0}"
        ]
        mob_5 = TexMobject(*formula_2_partital)
        mob_5.set_color_by_tex("\\mathbf{w}",GREEN)
        mob_5.set_color_by_tex("\\mathbf{R}",RED)

        
        formula_3 = [
            " \\mathbf{w}",    
            "=",
            "-",
            "\\lambda",
            "\\mathbf{R}",
            "^{-1}",
            "\\mathbf{a}(\\theta_0)"
        ]
        mob_6 = TexMobject(*formula_3)
        mob_6.set_color_by_tex("\\mathbf{w}",GREEN)
        mob_6.set_color_by_tex("\\mathbf{R}",RED)

        dark_bg = Rectangle(width = 2*FRAME_X_RADIUS, height = 2*FRAME_Y_RADIUS, fill_color = BLACK, fill_opacity = 0.7, color = BLACK)
        framebox3 = SurroundingRectangle(mob_2)
        constrain_formula = VGroup(*mob_2_copy[1:6])

        #expression of lambda
        formula_4 = [
            "(",
            "-",
            "\\lambda",
            "\\mathbf{R}",
            "^{-1}",
            "\\mathbf{a}(\\theta_0)",
            ")",
            "^{H}",
            "\\mathbf{a}(\\theta_{0})",
            "=",
            "1"
        ]
        mob_7 = TexMobject(*formula_4)
        mob_7.set_color_by_tex("\\mathbf{w}",GREEN)
        mob_7.set_color_by_tex("\\mathbf{R}",RED)

        formula_5 = [
            "-",
            "\\lambda",
            "\\mathbf{a}^{H}(\\theta_0)",
            "\\mathbf{R}",
            "^{-1}",
            "\\mathbf{a}(\\theta_{0})",
            "=",
            "1"
        ]
        mob_8 = TexMobject(*formula_5)
        mob_8.set_color_by_tex("\\mathbf{w}",GREEN)
        mob_8.set_color_by_tex("\\mathbf{R}",RED)

        formula_6 = [
            "\\lambda",
            "=",
            "{-",
            "1",
            "\\over",
            "\\mathbf{a}^{H}(\\theta_0)",
            "\\mathbf{R}",
            "^{-1}",
            "\\mathbf{a}(\\theta_{0})}",
        ]
        mob_9 = TexMobject(*formula_6)
        mob_9.set_color_by_tex("\\mathbf{w}",GREEN)
        mob_9.set_color_by_tex("\\mathbf{R}",RED)

        formula_7 = [
            "\\mathbf{W}_{MVDR}", 
            "=",
            "{-",
            "\\mathbf{a}(\\theta_{0})",
            "\\over",
            "\\mathbf{a}^{H}(\\theta_0)",
            "\\mathbf{R}",
            "^{-1}",
            "\\mathbf{a}(\\theta_{0})}",
        ]
        mob_10 = TexMobject(*formula_7)
        mob_10.set_color_by_tex("\\mathbf{R}",RED)



        self.play(Write(mob_1))
        self.wait()
        self.play(Write(mob_2))
        self.play(GrowFromCenter(brace_1),GrowFromCenter(brace_2_text))
        self.wait()
        self.play(Transform(brace_2_text,brace_1_text))
        self.wait()
        self.play(GrowFromCenter(brace_2),GrowFromCenter(brace_3_text))
        self.wait()
        self.play(ShowCreationThenFadeAround(framebox1),ShowCreationThenFadeAround(framebox2))
        self.wait()

        self.play(
            *[FadeOut(i) for i in [brace_1,brace_2,brace_3_text,brace_2_text,mob_1.remove(*corres_ele1_in_formula),mob_2.remove(*corres_ele1_in_formula)]],
            Write(rest_ele_in_formula), #write the rest of the element
            *[ReplacementTransform(i.copy(),j) for i,j in zip(corres_ele1_in_formula,corres_ele2_in_formula)], 
        )
        self.wait()
        self.clear()
        self.play(
            TransformFromCopy(mob_3_copy,mob_4),
            mob_3_copy.to_corner,UP+LEFT,
            mob_3_copy.shift,2*LEFT,
            mob_3_copy.scale,0.5
        )
        self.wait()
        #self.clear()
        self.play(
            *[ReplacementTransform(i.copy(),j) for i,j in zip(mob_4[10:],mob_5)],
            mob_4.move_to,mob_3_copy.get_left()+DOWN+RIGHT*mob_4.get_width()/4,
            mob_4.scale,0.5
        )
        self.wait()
        self.play(
            *[ReplacementTransform(mob_5[i],mob_6[j],path_arc = 2) for i,j in zip([0,1,2,3,4,5,6],[4,0,2,3,6,1,5])]
        )
        self.play(FadeIn(dark_bg))
        self.wait()
        self.play(Write(mob_1_copy),Write(mob_2_copy))
        self.wait()
        self.play(ShowCreationThenFadeAround(framebox3))
        self.wait()
        mob_6_copy = mob_6.copy()
        self.play(
            TransformFromCopy(mob_6.copy(),mob_6_copy.scale(0.5).move_to(mob_4.get_left()+DOWN+RIGHT*mob_6.get_width()/4)),
            *[FadeOut(i) for i in [dark_bg,mob_1_copy,mob_2_copy,mob_6]],
            *[ReplacementTransform(mob_6[i],mob_7[j]) for i,j in zip([2,3,4,5,6],[1,2,3,4,5])],
            *[ReplacementTransform(constrain_formula[i],mob_7[j]) for i,j in zip([1,2,3,4],[7,8,9,10])],
            *[Write(i) for i in [mob_7[0],mob_7[6]]],
        )
        self.wait()
        self.play(
            mob_7.move_to,mob_6_copy.get_left()+DOWN+RIGHT*mob_7.get_width()/4,
            mob_7.scale,0.5,
            *[ReplacementTransform(mob_7.copy()[i],mob_8[j]) for i,j in zip([1,2,3,4,5,7,8,9,10],[0,1,3,4,2,2,5,6,7])],
        )
        self.wait()
        self.play(
            mob_8.move_to,mob_7.get_left()+DOWN+RIGHT*mob_8.get_width()/4,
            mob_8.scale,0.5,
            *[ReplacementTransform(mob_8.copy()[i],mob_9[j] ,path_arc = 2) for i,j in zip([0,1,2,3,4,5,6,7],[2,0,5,6,7,8,1,3])],
            FadeIn(mob_9[4])
        )
        self.wait()
        self.play(
            TransformFromCopy(mob_9,mob_10),
            mob_9.move_to,mob_8.get_left()+DOWN+RIGHT*mob_9.get_width()/4,
            mob_9.scale,0.5,
        )

class testscene(Scene):
    def construct(self):
        mvdr_1 = MVDR(theta0 = 20, theta1 =55, normal = 4)
        plot_1 = ParametricFunction(
            lambda t: np.array([
                3.87*t,
                (mvdr_1.y.__call__(degrees(t))),
                0
            ]),
            color = RED,
            t_min = -pi/2 + 0.02,
            t_max = pi/2 - 0.02,
            step_size = 0.01
        ).shift([0,-3.5,0])
        print("w\n")
        print(mvdr_1.w)
        self.play(
            ShowCreation(plot_1)
        )
        self.wait()
        self.clear()
        mvdr_1.w.fill(0)
        mvdr_1.w[2] = 1
        new_y_1 = mvdr_1.generate_funcy(mvdr_1.w)
        plot_2 = ParametricFunction(
            lambda t: np.array([
                3.87*t,
                (new_y_1.__call__(degrees(t))),
                0
            ]),
            color = RED,
            t_min = -pi/2 + 0.02,
            t_max = pi/2 - 0.02,
            step_size = 0.01
        ).shift([0,-3.5,0])
        
        mvdr_1.w.fill(0)
        mvdr_1.w[4] = 1
        mvdr_1.w[2] = 0.5
        new_y_2 = mvdr_1.generate_funcy(mvdr_1.w)
        plot_3 = ParametricFunction(
            lambda t: np.array([
                3.87*t,
                (new_y_2.__call__(degrees(t))),
                0
            ]),
            color = RED,
            t_min = -pi/2 + 0.02,
            t_max = pi/2 - 0.02,
            step_size = 0.01
        ).shift([0,-3.5,0])

        print("w\n")
        print(mvdr_1.w)
        self.play(
            ShowCreation(plot_2)
        )
        self.wait()
        self.play(
            ReplacementTransform(plot_2,plot_3)
        )



