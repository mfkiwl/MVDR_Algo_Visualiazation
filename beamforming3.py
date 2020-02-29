from manimlib.imports import *
from math import *
from scipy.interpolate import interp1d
from manimlib.utils.config_ops import digest_config
from manimlib.once_useful_constructs.light import *

class Animation_of_demo_1(Scene):
    CONFIG={
		"camera_config": {
			"background_opacity": 0.6
		},
	}
    def get_sine_wave(self,dx=0,A = 1,w = 1):
        return FunctionGraph(
            lambda x: A*np.sin((w*x+dx)),
            x_min=-1.5,x_max=1.5,
            color = YELLOW
        )
    

    def construct(self):
        num_plane = NumberPlane()

        mic_array_size = 1.5
        plate = Circle(radius = 2, fill_opacity = 0.2, fill_color = WHITE, color = GREY)
        mic_0 = Dot(point = [mic_array_size, 0, 0], color = YELLOW)
        mic_1 = Dot(point = [0.5 * mic_array_size, 1.732 * 0.5 * mic_array_size, 0], color = YELLOW)
        mic_2 = Dot(point = [0.5 * mic_array_size, -1.732 * 0.5 * mic_array_size, 0], color = YELLOW)
        mic_3 = Dot(point = [-mic_array_size, 0, 0], color = YELLOW)
        mic_4 = Dot(point = [-0.5 * mic_array_size, -1.732 * 0.5 * mic_array_size, 0], color = YELLOW)
        mic_5 = Dot(point = [-0.5 * mic_array_size, 1.732 * 0.5 * mic_array_size, 0], color = YELLOW)  #hexagon shape
        mic_array = [plate,mic_0, mic_1, mic_2, mic_3, mic_4, mic_5]


        wave_diaplay = Rectangle(height = 2.5, width =3.5, color = GREEN, fill_color = GREEN, fill_opacity = 0.3).move_to([-4.5,2,0])
        responce_line = DashedLine(start = [-6.25,3,0], end = [-2.75,3,0] )
        text_responce = TextMobject("threshold").scale(0.5).next_to(responce_line,RIGHT)
        sine_function_1 = self.get_sine_wave(A = 1, w = 10)
        sine_function_2 = self.get_sine_wave(A = 0.2, w = 10)
        sine_function_3 = self.get_sine_wave(A = 0.5, w = 10)
        

        sine_function_1.move_to(wave_diaplay)
        sine_function_2.move_to(wave_diaplay)
        sine_function_3.move_to(wave_diaplay)

        arrow_1 = Arrow(start = [-1.5, 1.5, 0], end = [-2.75, 2, 0])
        arrow_2 = Arrow(start = [-4.5, 0.75, 0], end = [-4.5, -1, 0])

        judge_box = Rectangle(height = 2, width = 3, color = WHITE, fill_color = BLACK, fill_opacity = 1).move_to([-4.5, -2, 0])
        text_yes = TextMobject("Responce", color = GREEN).scale(0.8).move_to(judge_box)
        text_no = TextMobject("No Responce", color = RED).scale(0.8).move_to(judge_box)
        text_show = text_yes.deepcopy()
        def lobe_function(x):
            if x >= -90 and x < -30:
                return -(1/2700) * (x + 30) * (x + 90)
            elif x >= -30 and x < 30:
                return -(4/900) * (x - 30) * (x + 30)
            elif x >= 30 and x < 90:
                return -(1/2700) * (x - 30) * (x - 90)
            else:
                return 0
        lobe_graph = ParametricFunction(
            lambda t: np.array(
                [
                    np.cos(np.radians(t))*lobe_function(t),
                    np.sin(np.radians(t))*lobe_function(t),
                    0
                ]
            ),
            color = ORANGE,
            t_min = -90,
            t_max = 90,
            step_size = 0.1
        )

        
        voice_wave = []
        for i in range(0,5):
            voice_wave.append(
                Sector(angle = pi/4, start_angle = 7*pi/8, arc_center = [5.5,0,0],outer_radius = 0.5*(1+i), color = YELLOW, fill_color = YELLOW, fill_opacity = 0.6/(1+i))
            )
        svg_speakingman = SVGMobject("/Users/neoncloud/Documents/GitHub/manim-0.1.10/study/speaking2.svg").scale(0.3).move_to([6,0,0]).flip(axis = UP)
        voice_wave_group = VGroup(*voice_wave, svg_speakingman) 

        
        text_1 = TextMobject("Mic array").next_to(plate,UP)
        text_2 = TextMobject("Mainlobe").next_to(lobe_graph,UP).shift(RIGHT).scale(0.8)

        self.play(
            ShowCreation(num_plane)
        )
        self.wait()
        self.play(
            AnimationGroup(
                *[ShowCreation(i) for i in mic_array]
            ),
            lag_ratio = 0.1
        )
        self.wait()
        self.play(GrowFromEdge(text_1, DOWN))
        self.wait()
        self.play(
            ShowCreation(lobe_graph)
            ,run_time = 2
        )
        self.wait()
        self.play(GrowFromEdge(text_2,LEFT))
        self.play(
            ShowCreation(svg_speakingman),
            ShowCreation(wave_diaplay),
            ShowCreation(responce_line),
            ShowCreation(arrow_1),
            ShowCreation(arrow_2),
            ShowCreation(judge_box),
            Write(text_responce)
        )
        self.wait()
        self.play(
            AnimationGroup(
                *[FadeIn(i, rate_func = there_and_back) for i in deepcopy(voice_wave)],
                ShowCreationThenDestruction(sine_function_1),
                Write(text_show),
                lag_ratio = 0.1
            ),
            run_time = 1.6,
            
        )
        self.remove(*voice_wave)
        self.play(
            AnimationGroup(
                *[FadeIn(i, rate_func = there_and_back) for i in deepcopy(voice_wave)],
                ShowCreationThenDestruction(sine_function_1),
                lag_ratio = 0.1
            ),
            run_time = 1.6
        )
        self.remove(*voice_wave)
        self.wait()
        self.play(
            Rotate(svg_speakingman, pi/6, about_point = ORIGIN)
        )
        for i in voice_wave:
            i.rotate_about_origin(pi/6)
            
        self.play(
            AnimationGroup(
                *[FadeIn(i, rate_func = there_and_back) for i in deepcopy(voice_wave)],
                ShowCreationThenDestruction(sine_function_2),
                Transform(text_show,text_no),
                lag_ratio = 0.1
            ),
            run_time = 1.6
        )
        self.remove(*voice_wave)
        self.play(
            AnimationGroup(
                *[FadeIn(i, rate_func = there_and_back) for i in deepcopy(voice_wave)],
                ShowCreationThenDestruction(sine_function_2),
                lag_ratio = 0.1
            ),
            run_time = 1.6
        )
        self.remove(*voice_wave)
        self.wait()
        self.play(
            Rotate(svg_speakingman, -pi/4, about_point = ORIGIN)
        )
        for i in voice_wave:
            i.rotate_about_origin(-pi/4)
            
        self.play(
            AnimationGroup(
                *[FadeIn(i, rate_func = there_and_back) for i in deepcopy(voice_wave)],
                ShowCreationThenDestruction(sine_function_3),
                lag_ratio = 0.1
            ),
            run_time = 1.6
        )
        self.remove(*voice_wave)
        self.play(
            AnimationGroup(
                *[FadeIn(i, rate_func = there_and_back) for i in deepcopy(voice_wave)],
                ShowCreationThenDestruction(sine_function_3),
                lag_ratio = 0.1
            ),
            run_time = 1.6
        )
        self.remove(*voice_wave)
        self.wait()
        self.play(
            Rotate(svg_speakingman, pi/12, about_point = ORIGIN)
        )
        for i in voice_wave:
            i.rotate_about_origin(pi/12)
            
        self.play(
            AnimationGroup(
                *[FadeIn(i, rate_func = there_and_back) for i in deepcopy(voice_wave)],
                ShowCreationThenDestruction(sine_function_1),
                Transform(text_show,text_yes),
                lag_ratio = 0.1
            ),
            run_time = 1.6
        )
        self.remove(*voice_wave)
        self.play(
            AnimationGroup(
                *[FadeIn(i, rate_func = there_and_back) for i in deepcopy(voice_wave)],
                ShowCreationThenDestruction(sine_function_1),
                lag_ratio = 0.1
            ),
            run_time = 1.6
        )
        self.remove(*voice_wave)
        self.wait()

class Animation_of_demo_2(Scene):
    def construct(self):
        
        svg_speakingman = SVGMobject("/Users/neoncloud/Documents/GitHub/manim-0.1.10/study/speaking3.svg", stroke_width=2.0, fill_opacity = 0).flip(axis = UP).scale(1.2)
        mic_0 = Dot(point = [0, 0, 0], color = YELLOW)
        mic_1 = Dot(point = [-0.5, -0.5, 0], color = YELLOW)

        def lobe_function(x):
            if x >= -90 and x < -30:
                return -(1/2700) * (x + 30) * (x + 90)
            elif x >= -30 and x < 30:
                return -(2/900) * (x - 30) * (x + 30)
            elif x >= 30 and x < 90:
                return -(1/2700) * (x - 30) * (x - 90)
            else:
                return 0
        lobe_graph = ParametricFunction(
            lambda t: np.array(
                [
                    np.cos(np.radians(t))*lobe_function(t),
                    np.sin(np.radians(t))*lobe_function(t),
                    0
                ]
            ),
            color = ORANGE,
            t_min = -90,
            t_max = 90,
            step_size = 0.1
        )
        lobe_graph.rotate(-3*pi/4, about_point = ORIGIN)
        text_1 = TextMobject("Mainlobe").next_to(lobe_graph,LEFT).scale(0.8)

        voice_wave = []
        for i in range(0,5):
            voice_wave.append(
                Sector(angle = pi/4, start_angle = -7*pi/8, arc_center = [-0.5,-0.5,0],outer_radius = 0.5*(1+i), color = YELLOW, fill_color = YELLOW, fill_opacity = 0.6/(1+i))
            )

        wave = []
        for i in range(0,5):
            wave.append(Arc(angle = 2*pi, arc_center = [8,0,0], radius = 0.01, fill_color = YELLOW, fill_opacity = 0.1, color = YELLOW, opacity = 0.1))

        wave_target = []
        for i in range(0,5):
            wave_target.append(Arc(angle = 2*pi, arc_center = [8,0,0], radius = 7, fill_color = YELLOW, fill_opacity = 0, color = WHITE, opacity = 0, stroke_width = 0))
        
        self.play(Write(svg_speakingman))
        self.wait()
        self.play(
            GrowFromCenter(mic_0),
            GrowFromCenter(mic_1)
        )
        self.play(Flash(mic_0),Flash(mic_1))
        self.wait()
        self.play(ShowCreation(lobe_graph),Write(text_1))
        self.wait()
        self.play(
            AnimationGroup(
                *[FadeIn(i, rate_func = there_and_back) for i in deepcopy(voice_wave)],
                lag_ratio = 0.1
            ),
            run_time = 1.6
        )
        self.remove(*voice_wave)
        self.play(
            AnimationGroup(
                *[FadeIn(i, rate_func = there_and_back) for i in deepcopy(voice_wave)],
                lag_ratio = 0.1
            ),
            run_time = 1.6
        )
        self.remove(*voice_wave)
        self.wait()

        self.add(*wave)
        self.play(
            AnimationGroup(
                *[
                    Transform(i.copy(),j.copy()) for i,j in zip(wave,wave_target)
                ],
                lag_ratio = 0.1
            ),
            run_time = 3
        )
        self.remove(*wave,*wave_target)
        self.play(
            AnimationGroup(
                *[
                    Transform(i.copy(),j.copy()) for i,j in zip(wave,wave_target)
                ],
                lag_ratio = 0.1
            ),
            run_time = 3
        )

        
class Animation_of_demo_3(Scene):
    

    def construct(self):
        num_plane = NumberPlane()

        mic_array_size = 1
        plate = Rectangle(width = 4, height = 2, fill_opacity = 0.2, fill_color = WHITE, color = GREY)
        bar = Rectangle(width = 3.5, height = 0.5, fill_opacity = 0.2, fill_color = BLUE, color = BLUE)
        mic_0 = Dot(point = [mic_array_size, 0, 0], color = YELLOW)
        mic_1 = Dot(point = [-mic_array_size, 0, 0], color = YELLOW)
        mic_array = [plate,bar,mic_0,mic_1]
        

        def lobe_function(x):
            if (x >= -30 and x < 30):
                return -(4/900) * (x - 30) * (x + 30)
            elif (x >= 150 and x < 210):
                return -(4/900) * ((x-180) - 30) * ((x-180) + 30)
            else:
                return 0

        lobe_graph = ParametricFunction(
            lambda t: np.array(
                [
                    np.cos(np.radians(t))*lobe_function(t),
                    np.sin(np.radians(t))*lobe_function(t),
                    0
                ]
            ),
            color = ORANGE,
            t_min = -30,
            t_max = 330,
            step_size = 0.1
        )

        text_1 = TextMobject("Mic array").next_to(plate,UP)
        text_2 = TextMobject("Mainlobe").next_to(lobe_graph,RIGHT).shift(DOWN+2*LEFT).scale(0.8)

        svg_speakingman_1 = SVGMobject("/Users/neoncloud/Documents/GitHub/manim-0.1.10/study/speaking2.svg").scale(0.3).move_to([6,0,0]).flip(axis = UP)
        svg_speakingman_2 = SVGMobject("/Users/neoncloud/Documents/GitHub/manim-0.1.10/study/speaking2.svg").scale(0.3).move_to([-6,0,0])
        svg_pocessor = SVGMobject("/Users/neoncloud/Documents/GitHub/manim-0.1.10/study/Processor.svg").scale(0.5).move_to([3,2,0])
        text_speaker_1 = TextMobject("Speaker 1",color = GREEN).next_to(svg_speakingman_1,DOWN).scale(0.5)
        text_speaker_2 = TextMobject("Speaker 2",color = GREEN).next_to(svg_speakingman_2,DOWN).scale(0.5)

        voice_wave_1 = []
        for i in range(0,5):
            voice_wave_1.append(
                Sector(angle = pi/4, start_angle = 7*pi/8, arc_center = [5.5,0,0],outer_radius = 0.5*(1+i), color = YELLOW, fill_color = YELLOW, fill_opacity = 0.6/(1+i))
            )
        voice_wave_2 = []
        for i in range(0,5):
            voice_wave_2.append(
                Sector(angle = pi/4, start_angle = -pi/8, arc_center = [-5.5,0,0],outer_radius = 0.5*(1+i), color = YELLOW, fill_color = YELLOW, fill_opacity = 0.6/(1+i))
            )
            
        self.play(
            ShowCreation(num_plane)
        )
        self.wait()
        self.play(
            Write(text_1),
            AnimationGroup(
                *[ShowCreation(i) for i in mic_array]
            ),
            lag_ratio = 0.1
        )
        self.wait()
        self.play(
            Write(text_2),
            ShowCreation(lobe_graph)
        )
        self.wait()
        self.play(
            GrowFromCenter(svg_speakingman_1),
            GrowFromCenter(svg_speakingman_2),
            Write(text_speaker_1),
            Write(text_speaker_2),
        )
        self.wait()
        self.play(
            AnimationGroup(
                *[FadeIn(i, rate_func = there_and_back) for i in deepcopy(voice_wave_1)],
                lag_ratio = 0.1
            ),
            AnimationGroup(
                *[FadeIn(i, rate_func = there_and_back) for i in deepcopy(voice_wave_2)],
                lag_ratio = 0.1
            ),
            run_time = 1.6
        )
        self.remove(*voice_wave_1,*voice_wave_2)
        self.play(
            AnimationGroup(
                *[FadeIn(i, rate_func = there_and_back) for i in deepcopy(voice_wave_1)],
                lag_ratio = 0.1
            ),
            AnimationGroup(
                *[FadeIn(i, rate_func = there_and_back) for i in deepcopy(voice_wave_2)],
                lag_ratio = 0.1
            ),
            run_time = 1.6
        )
        self.remove(*voice_wave_1,*voice_wave_2)
        self.wait()
        self.play(Flash(mic_0),Flash(mic_1))
        self.wait()
        self.play(
            GrowFromPoint(svg_pocessor,point = ORIGIN, rate_func = there_and_back_with_pause), run_time = 2.5
        )
        self.wait()

class small_1(Scene):
    def construct(self):
        num_plane = NumberPlane()
        dot_mic = Dot(point=ORIGIN, color = YELLOW, radius = 0.2)
        dot_speaker = Dot(point=[4,0,0], color = BLUE, radius = 0.2).rotate_about_origin(pi/4)
        text_recever = TextMobject("Receiver").next_to(dot_mic)
        text_speaker = TextMobject("Speaker")

        voice_wave = []
        for i in range(0,6):
            voice_wave.append(
                Sector(angle = pi/4, start_angle = -7*pi/8, arc_center = dot_speaker.get_center(),outer_radius = 0.5*(1+i), color = GREEN, fill_color = GREEN, fill_opacity = 0.6/(1+i))
            )

        def update_func(obj):
            obj.next_to(dot_speaker)

        text_speaker.add_updater(update_func)

        def lobe_function(x):
            if x >= -90 and x < -30:
                return -(1/2700) * (x + 30) * (x + 90)
            elif x >= -30 and x < 30:
                return -(4/900) * (x - 30) * (x + 30)
            elif x >= 30 and x < 90:
                return -(1/2700) * (x - 30) * (x - 90)
            else:
                return 0
        lobe_graph = ParametricFunction(
            lambda t: np.array(
                [
                    np.cos(np.radians(t))*lobe_function(t),
                    np.sin(np.radians(t))*lobe_function(t),
                    0
                ]
            ),
            color = ORANGE,
            t_min = -90,
            t_max = 90,
            step_size = 0.1,
            stroke_opacity = 0.7
        )

        self.play(
            ShowCreation(num_plane),
            GrowFromCenter(dot_mic),
            GrowFromCenter(dot_speaker),
            GrowFromCenter(lobe_graph),
            Write(text_speaker),
            Write(text_recever)
        )
        self.wait()
        self.play(
            AnimationGroup(
                *[FadeIn(i, rate_func = there_and_back) for i in deepcopy(voice_wave)],
                lag_ratio = 0.1
            ),
            run_time = 1.6
        )
        self.remove(*voice_wave)
        for i in voice_wave:
            i.rotate_about_origin(-pi/12)
        self.wait()
        self.play(
            Rotate(dot_speaker,angle = -pi/12, about_point = ORIGIN),
            UpdateFromFunc(text_speaker,update_func)
        )
        self.play(
            AnimationGroup(
                *[FadeIn(i, rate_func = there_and_back) for i in deepcopy(voice_wave)],
                lag_ratio = 0.1
            ),
            run_time = 1.6
        )
        self.remove(*voice_wave)
        for i in voice_wave:
            i.rotate_about_origin(-pi/12)
        self.wait()
        self.play(
            Rotate(dot_speaker,angle = -pi/12, about_point = ORIGIN),
            UpdateFromFunc(text_speaker,update_func)
        )
        self.play(
            AnimationGroup(
                *[FadeIn(i, rate_func = there_and_back) for i in deepcopy(voice_wave)],
                lag_ratio = 0.1
            ),
            run_time = 1.6
        )
        self.remove(*voice_wave)
        for i in voice_wave:
            i.rotate_about_origin(-pi/12)
        self.wait()
        self.play(
            Rotate(dot_speaker,angle = -pi/12, about_point = ORIGIN),
            UpdateFromFunc(text_speaker,update_func)
        )
        self.play(
            AnimationGroup(
                *[FadeIn(i, rate_func = there_and_back) for i in deepcopy(voice_wave)],
                Flash(dot_mic,flash_radius=0.5,line_length=0.3),
                lag_ratio = 0.1
            ),
            run_time = 1.6
        )
        self.remove(*voice_wave)
        for i in voice_wave:
            i.rotate_about_origin(-pi/12)
        self.wait()
        self.play(
            Rotate(dot_speaker,angle = -pi/12, about_point = ORIGIN),
            UpdateFromFunc(text_speaker,update_func)
        )
        self.play(
            AnimationGroup(
                *[FadeIn(i, rate_func = there_and_back) for i in deepcopy(voice_wave)],
                lag_ratio = 0.1
            ),
            run_time = 1.6
        )
        self.remove(*voice_wave)
        for i in voice_wave:
            i.rotate_about_origin(-pi/12)
        self.wait()
        

class small_2(Scene):
    def construct(self):
        num_plane = NumberPlane(
            CONFIG = {
            "axis_config": {
                "stroke_opacity": 0.3}
                }
        )
        dot_mic = Dot(point=ORIGIN, color = YELLOW, radius = 0.2)
        #dot_speaker_1 = Dot(point=[4,0,0], color = BLUE, radius = 0.2)
        dot_speaker_2 = Dot(point=[0,3,0], color = BLUE, radius = 0.2)
        lable_1 = TextMobject("Speaker").next_to(dot_mic,RIGHT)
        lable_2 = TextMobject("Noise").next_to(dot_speaker_2,RIGHT)
        wave = []
        for i in range(0,5):
            wave.append(Arc(angle = 2*pi, arc_center = [0,3,0], radius = 0.01, fill_color = YELLOW, fill_opacity = 0.1, color = YELLOW, opacity = 0.1))

        wave_target = []
        for i in range(0,5):
            wave_target.append(Arc(angle = 2*pi, arc_center = [0,3,0], radius = 7, fill_color = YELLOW, fill_opacity = 0, color = WHITE, opacity = 0, stroke_width = 0))
        

        voice_wave_1 = []
        for i in range(0,6):
            voice_wave_1.append(
                Sector(angle = pi/4, start_angle = -5*pi/8, arc_center = [0,0,0],outer_radius = 0.5*(1+i), color = GREEN, fill_color = GREEN, fill_opacity = 0.6/(1+i)).scale(1.2)
            )

        def lobe_function(x):
            if x >= -90 and x < -30:
                return -(1/2700) * (x + 30) * (x + 90)
            elif x >= -30 and x < 30:
                return -(2/900) * (x - 30) * (x + 30)
            elif x >= 30 and x < 90:
                return -(1/2700) * (x - 30) * (x - 90)
            else:
                return 0
        lobe_graph = ParametricFunction(
            lambda t: np.array(
                [
                    np.cos(np.radians(t))*lobe_function(t),
                    np.sin(np.radians(t))*lobe_function(t),
                    0
                ]
            ),
            color = ORANGE,
            t_min = -90,
            t_max = 90,
            step_size = 0.1
        )
        lobe_graph.rotate(-pi/2, about_point = ORIGIN)
        #def updater_1(obj):
        #    obj.next_to(dot_speaker_1,RIGHT)

        #lable_1.add_updater(updater_1)

        self.play(
            ShowCreation(num_plane),
            GrowFromCenter(dot_mic),
            GrowFromCenter(lobe_graph),
            GrowFromCenter(dot_speaker_2),
            Write(lable_1),
            Write(lable_2)
        )
        self.wait()
        """ self.play(
            Rotate(dot_speaker_1,angle = pi, about_point = ORIGIN),
            UpdateFromFunc(lable_1,updater_1)
        )
        self.wait() """

        self.add(*wave)
        self.play(
            AnimationGroup(
                *[FadeIn(i, rate_func = there_and_back) for i in deepcopy(voice_wave_1)],
                lag_ratio = 0.1
            ),
            AnimationGroup(
                *[
                    Transform(i.copy(),j.copy()) for i,j in zip(wave,wave_target)
                ],
                lag_ratio = 0.1
            ),
            run_time = 3
        )
        self.remove(*wave,*wave_target,*voice_wave_1)
        self.play(
            AnimationGroup(
                *[FadeIn(i, rate_func = there_and_back) for i in deepcopy(voice_wave_1)],
                lag_ratio = 0.1
            ),
            AnimationGroup(
                *[
                    Transform(i.copy(),j.copy()) for i,j in zip(wave,wave_target)
                ],
                lag_ratio = 0.1
            ),
            run_time = 3
        )
        self.add(*wave)
        self.play(
            AnimationGroup(
                *[FadeIn(i, rate_func = there_and_back) for i in deepcopy(voice_wave_1)],
                lag_ratio = 0.1
            ),
            AnimationGroup(
                *[
                    Transform(i.copy(),j.copy()) for i,j in zip(wave,wave_target)
                ],
                lag_ratio = 0.1
            ),
            run_time = 3
        )
        
        self.remove(*wave,*wave_target,*voice_wave_1)
        self.play(
            AnimationGroup(
                *[FadeIn(i, rate_func = there_and_back) for i in deepcopy(voice_wave_1)],
                lag_ratio = 0.1
            ),
            AnimationGroup(
                *[
                    Transform(i.copy(),j.copy()) for i,j in zip(wave,wave_target)
                ],
                lag_ratio = 0.1
            ),
            run_time = 3
        )
class end(Scene):
    def construct(self):
        text = TextMobject("Thank you for your watching!")
        self.play(Write(text))