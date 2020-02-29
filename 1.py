from manimlib.imports import *

class Test_scene(Scene):
    def construct(self):
        #######Code#######
        #Making text
        first_line = TextMobject("Manim is fun")
        second_line = TextMobject("and useful")
        final_line = TextMobject("Hope you like it too!", color=BLUE)
        color_final_line = TextMobject("Hope you like it too!")

        #Coloring
        color_final_line.set_color_by_gradient(BLUE,PURPLE)

        #Position text
        second_line.next_to(first_line, DOWN)

        #Showing text
        self.wait(1)
        self.play(Write(first_line), Write(second_line))
        self.wait(1)
        self.play(FadeOut(second_line), ReplacementTransform(first_line, final_line))
        self.wait(1)
        self.play(Transform(final_line, color_final_line))
        self.wait(2)

class concurrent(Scene):
    def construct(self):
        dot1 = Dot()
        dot2 = Dot()
        dot2.shift(UP)
        dot3 = Dot()
        dot3.shift(DOWN)
 
		# 单个动画的演示
        self.play(Write(dot1))
        # 多个动画演示
        self.play(*[
            Transform(i.copy(),j) for i,j in zip([dot1,dot1],[dot2,dot3])
            ])# 故意使用i,j是为了显示zip的使用

        self.wait()
class ChangeColorAndSizeAnimation(Scene):
	def construct(self):
         text = TextMobject("Text")
         self.play(Write(text))

         text.generate_target()
         text.target = TextMobject("Target")
         text.target.set_color(RED)
         text.target.scale(2)
         text.target.shift(LEFT)

         self.play(MoveToTarget(text),run_time = 2)
         self.wait()

class SurfacesAnimation(ThreeDScene):
    def construct(self):
        axes = ThreeDAxes()
        cylinder = ParametricSurface(
            lambda u, v: np.array([
                np.cos(TAU * v),
                np.sin(TAU * v),
                2 * (1 - u)
            ]),
            resolution=(6, 32)).fade(0.5) #Resolution of the surfaces

        paraboloid = ParametricSurface(
            lambda u, v: np.array([
                np.cos(v)*u,
                np.sin(v)*u,
                u**2
            ]),v_max=TAU,
            checkerboard_colors=[PURPLE_D, PURPLE_E],
            resolution=(10, 32)).scale(2)

        para_hyp = ParametricSurface(
            lambda u, v: np.array([
                u,
                v,
                u**2-v**2
            ]),v_min=-2,v_max=2,u_min=-2,u_max=2,checkerboard_colors=[BLUE_D, BLUE_E],
            resolution=(15, 32)).scale(1)

        cone = ParametricSurface(
            lambda u, v: np.array([
                u*np.cos(v),
                u*np.sin(v),
                u
            ]),v_min=0,v_max=TAU,u_min=-2,u_max=2,checkerboard_colors=[GREEN_D, GREEN_E],
            resolution=(15, 32)).scale(1)

        hip_one_side = ParametricSurface(
            lambda u, v: np.array([
                np.cosh(u)*np.cos(v),
                np.cosh(u)*np.sin(v),
                np.sinh(u)
            ]),v_min=0,v_max=TAU,u_min=-2,u_max=2,checkerboard_colors=[YELLOW_D, YELLOW_E],
            resolution=(15, 32))

        ellipsoid=ParametricSurface(
            lambda u, v: np.array([
                1*np.cos(u)*np.cos(v),
                2*np.cos(u)*np.sin(v),
                0.5*np.sin(u)
            ]),v_min=0,v_max=TAU,u_min=-PI/2,u_max=PI/2,checkerboard_colors=[TEAL_D, TEAL_E],
            resolution=(15, 32)).scale(2)

        sphere = ParametricSurface(
            lambda u, v: np.array([
                1.5*np.cos(u)*np.cos(v),
                1.5*np.cos(u)*np.sin(v),
                1.5*np.sin(u)
            ]),v_min=0,v_max=TAU,u_min=-PI/2,u_max=PI/2,checkerboard_colors=[RED_D, RED_E],
            resolution=(15, 32)).scale(2)


        self.set_camera_orientation(phi=75 * DEGREES)
        self.begin_ambient_camera_rotation(rate=0.2)


        self.add(axes)
        self.play(Write(sphere))
        self.wait()
        self.play(ReplacementTransform(sphere,ellipsoid))
        self.wait()
        self.play(ReplacementTransform(ellipsoid,cone))
        self.wait()
        self.play(ReplacementTransform(cone,hip_one_side))
        self.wait()
        self.play(ReplacementTransform(hip_one_side,para_hyp))
        self.wait()
        self.play(ReplacementTransform(para_hyp,paraboloid))
        self.wait()
        self.play(ReplacementTransform(paraboloid,cylinder))
        self.wait()
        self.play(FadeOut(cylinder))


class SVGtest(Scene):
    def construct(self):
        svg_5g = SVGMobject("/Users/neoncloud/Documents/GitHub/manim-0.1.10/study/5G.svg").scale(0.5).move_to([-3,0,0])
        svg_5g[1].set_color(BLUE)
        svg_5g[0].set_color(RED)

        svg_wifi = SVGMobject("/Users/neoncloud/Documents/GitHub/manim-0.1.10/study/WiFi_Logo.svg").scale(0.5).move_to([0,0,0])
        svg_wifi[1].set_color(BLACK)
        svg_wifi[2].set_color(BLACK)
        svg_wifi[3].set_color(BLACK)

        svg_3dot = SVGMobject("/Users/neoncloud/Documents/GitHub/manim-0.1.10/study/3dot_1.svg").scale(0.1).move_to([3,0,0])

        self.play(
            Write(svg_5g)
        )
        self.play(
            Write(svg_wifi)
        )
        self.play(
            Write(svg_3dot)
        )