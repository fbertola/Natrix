from ctypes import c_float, sizeof, Structure
from pathlib import Path

import glfw
import numpy as np
from bgfx import (
    bgfx,
    ImGui,
    ImGuiExtra,
    BGFX_CLEAR_COLOR,
    BGFX_CLEAR_DEPTH,
    BGFX_RESET_VSYNC,
    BGFX_DEBUG_TEXT,
    as_void_ptr,
    ShaderType,
    load_shader,
    BGFX_STATE_WRITE_RGB,
    BGFX_STATE_WRITE_A,
    BGFX_STATE_DEFAULT,
    BGFX_TEXTURE_COMPUTE_WRITE,
    BGFX_RESET_HIDPI,
    BGFX_STATE_BLEND_ALPHA,
)
from loguru import logger

from demo.example_window import ExampleWindow
from demo.smooth_particles_area import SmoothParticlesArea
from demo.utils.imgui_utils import show_properties_dialog
from demo.utils.matrix_utils import look_at, proj
from natrix.core.fluid_simulator import FluidSimulator

logger.enable("bgfx")


class PosColorVertex(Structure):
    _fields_ = [
        ("m_x", c_float),
        ("m_y", c_float),
        ("m_z", c_float),
        ("m_u", c_float),
        ("m_v", c_float),
        ("m_w", c_float),
    ]


cube_vertices = (PosColorVertex * 4)(
    PosColorVertex(-1.0, 1.0, 0.0, -1.0, 1.0, 1.0),
    PosColorVertex(1.0, 1.0, 0.0, 1.0, 1.0, 1.0),
    PosColorVertex(-1.0, -1.0, 0.0, -1.0, -1.0, 1.0),
    PosColorVertex(1.0, -1.0, 0.0, 1.0, -1.0, 1.0),
)

cube_indices = np.array([0, 1, 2, 1, 2, 3], dtype=np.uint16)

root_path = Path(__file__).parent / "shaders"


class SimulationDemo(ExampleWindow):
    def __init__(self, width, height, title):
        super().__init__(width, height, title)

        self.init_conf = bgfx.Init()
        self.init_conf.debug = True
        self.init_conf.resolution.width = self.fb_width
        self.init_conf.resolution.height = self.fb_height
        self.init_conf.resolution.reset = BGFX_RESET_VSYNC | BGFX_RESET_HIDPI
        self.old_mouse_x = -1
        self.old_mouse_y = -1

        self.show_quiver_plot_overlay = ImGui.Bool(False)
        self.quiver_plot_resolution = ImGui.Int(2)
        self.quiver_plot_resolutions = (8.0, 16.0, 32.0, 64.0)

    def init(self, platform_data):
        self.init_conf.platform_data = platform_data
        bgfx.render_frame()
        bgfx.init(self.init_conf)
        bgfx.reset(
            self.fb_width,
            self.fb_height,
            BGFX_RESET_VSYNC | BGFX_RESET_HIDPI,
            self.init_conf.resolution.format,
        )

        bgfx.set_debug(BGFX_DEBUG_TEXT)
        bgfx.set_view_clear(0, BGFX_CLEAR_COLOR | BGFX_CLEAR_DEPTH, 0x0, 1.0, 0)

        self.vertex_layout = bgfx.VertexLayout()
        self.vertex_layout.begin().add(
            bgfx.Attrib.POSITION, 3, bgfx.AttribType.FLOAT
        ).add(bgfx.Attrib.TEXCOORD0, 3, bgfx.AttribType.FLOAT).end()

        self.fluid_simulator = FluidSimulator(
            self.width // 2, self.height // 2, self.vertex_layout
        )
        self.fluid_simulator.vorticity = 1.0
        self.fluid_simulator.viscosity = 0.5
        self.fluid_simulator.iterations = 50

        self.particle_area = SmoothParticlesArea(
            self.fb_width, self.fb_height, self.fluid_simulator, self.vertex_layout
        )
        self.particle_area.dissipation = 0.980

        # Create static vertex buffer
        vb_memory = bgfx.copy(as_void_ptr(cube_vertices), sizeof(PosColorVertex) * 4)
        self.vertex_buffer = bgfx.create_vertex_buffer(vb_memory, self.vertex_layout)

        ib_memory = bgfx.copy(as_void_ptr(cube_indices), cube_indices.nbytes)
        self.index_buffer = bgfx.create_index_buffer(ib_memory)

        self.output_texture = bgfx.create_texture2d(
            self.fb_width,
            self.fb_height,
            False,
            1,
            bgfx.TextureFormat.RGBA8,
            BGFX_TEXTURE_COMPUTE_WRITE,
        )

        self.texture_uniform = bgfx.create_uniform(
            "InputTexture", bgfx.UniformType.SAMPLER
        )
        self.window_size_uniform = bgfx.create_uniform(
            "WindowSize", bgfx.UniformType.VEC4
        )
        self.velocity_size_uniform = bgfx.create_uniform(
            "VelocitySize", bgfx.UniformType.VEC4
        )
        self.arrow_tile_size_uniform = bgfx.create_uniform(
            "ArrowTileSize", bgfx.UniformType.VEC4
        )

        # Create program from shaders.
        self.main_program = bgfx.create_program(
            load_shader(
                "demo.VertexShader.vert", ShaderType.VERTEX, root_path=root_path
            ),
            load_shader(
                "demo.FieldFragmentShader.frag",
                ShaderType.FRAGMENT,
                root_path=root_path,
            ),
            True,
        )
        self.quiver_program = bgfx.create_program(
            load_shader(
                "demo.VertexShader.vert", ShaderType.VERTEX, root_path=root_path
            ),
            load_shader(
                "demo.QuiverFragmentShader.frag",
                ShaderType.FRAGMENT,
                root_path=root_path,
            ),
            True,
        )
        self.cs_program = bgfx.create_program(
            load_shader(
                "demo.ComputeShader.comp", ShaderType.COMPUTE, root_path=root_path
            ),
            True,
        )

        ImGuiExtra.imgui_create(36.0)

    def shutdown(self):
        self.fluid_simulator.destroy()
        self.particle_area.destroy()

        ImGuiExtra.imgui_destroy()

        bgfx.destroy(self.index_buffer)
        bgfx.destroy(self.vertex_buffer)
        bgfx.destroy(self.output_texture)
        bgfx.destroy(self.texture_uniform)
        bgfx.destroy(self.window_size_uniform)
        bgfx.destroy(self.velocity_size_uniform)
        bgfx.destroy(self.arrow_tile_size_uniform)
        bgfx.destroy(self.main_program)
        bgfx.destroy(self.quiver_program)
        bgfx.destroy(self.cs_program)

        bgfx.shutdown()

    def update(self, dt):
        mouse_x, mouse_y, buttons_states = self.get_mouse_state()

        ImGuiExtra.imgui_begin_frame(
            int(mouse_x), int(mouse_y), buttons_states, 0, self.fb_width, self.fb_height
        )
        show_properties_dialog(self.fb_width, self.fb_height, self.hidpi)
        self._create_imgui_config_dialog()

        ImGuiExtra.imgui_end_frame()

        at = (c_float * 3)(*[0.0, 0.0, 0.0])
        eye = (c_float * 3)(*[0.0, 0.0, 10.0])
        up = (c_float * 3)(*[0.0, 1.0, 0.0])

        view = look_at(eye, at, up)
        projection = proj(11.4, 1, 0.1, 100.0)

        bgfx.set_view_rect(0, 0, 0, self.fb_width, self.fb_height)

        bgfx.set_view_transform(0, as_void_ptr(view), as_void_ptr(projection))

        bgfx.set_vertex_buffer(0, self.vertex_buffer, 0, 4)
        bgfx.set_index_buffer(self.index_buffer, 0, cube_indices.size)

        bgfx.set_state(BGFX_STATE_DEFAULT)
        bgfx.set_image(0, self.output_texture, 0, bgfx.Access.WRITE)

        self.fluid_simulator.add_circle_obstacle((0.5, 0.5), 40.0)

        self.fluid_simulator.update(dt)
        self.particle_area.update(dt)

        if glfw.get_mouse_button(self.window, glfw.MOUSE_BUTTON_LEFT) == glfw.PRESS:
            n_mouse_x, n_mouse_y = self._get_normalized_mouse_coords(mouse_x, mouse_y)
            n_old_mouse_x, n_old_mouse_y = self._get_normalized_mouse_coords(
                self.old_mouse_x, self.old_mouse_y
            )
            vel_x = n_mouse_x - n_old_mouse_x
            vel_y = n_mouse_y - n_old_mouse_y
            # if vel_x != 0 and vel_y != 0:
            self.fluid_simulator.add_velocity(
                (n_mouse_x, n_mouse_y), (vel_x * 10, vel_y * 10), 32.0
            )
            self.particle_area.add_particles((n_mouse_x, n_mouse_y), 200.0, 0.1)

        self.old_mouse_x = mouse_x
        self.old_mouse_y = mouse_y

        bgfx.set_state(
            0 | BGFX_STATE_WRITE_RGB | BGFX_STATE_WRITE_A | BGFX_STATE_BLEND_ALPHA
        )

        bgfx.dispatch(0, self.cs_program, self.fb_width // 16, self.fb_height // 16)
        bgfx.set_texture(0, self.texture_uniform, self.output_texture)
        bgfx.submit(0, self.main_program, 0, False)

        if self.show_quiver_plot_overlay.value:
            bgfx.set_buffer(
                1, self.fluid_simulator.get_velocity_buffer(), bgfx.Access.READ
            )
            bgfx.set_uniform(
                self.window_size_uniform,
                as_void_ptr((c_float * 2)(self.fb_width, self.fb_height)),
            )
            bgfx.set_uniform(
                self.velocity_size_uniform,
                as_void_ptr(
                    (c_float * 2)(
                        self.fluid_simulator.width, self.fluid_simulator.height
                    )
                ),
            )
            bgfx.set_uniform(
                self.arrow_tile_size_uniform,
                as_void_ptr(
                    (c_float * 1)(
                        self.quiver_plot_resolutions[self.quiver_plot_resolution.value]
                    )
                ),
            )
            bgfx.submit(0, self.quiver_program, 0, False)

        bgfx.frame()

    def resize(self):
        bgfx.reset(
            self.fb_width,
            self.fb_height,
            BGFX_RESET_VSYNC | BGFX_RESET_HIDPI,
            self.init_conf.resolution.format,
        )

    def _create_imgui_config_dialog(self):
        res_multiplier = 2 if self.hidpi else 1

        ImGui.set_next_window_pos(
            ImGui.Vec2(
                self.fb_width - self.fb_width / 4.1 - 20.0 * res_multiplier,
                20.0 * res_multiplier,
            ),
            ImGui.Condition.FirstUseEver,
        )
        ImGui.set_next_window_size(
            ImGui.Vec2(self.fb_width / 4.1, self.fb_height / 2.3),
            ImGui.Condition.FirstUseEver,
        )

        ImGui.begin("\uf013 Settings")

        vorticity = ImGui.Float(self.fluid_simulator.vorticity)
        viscosity = ImGui.Float(self.fluid_simulator.viscosity)
        speed = ImGui.Float(self.fluid_simulator.speed)
        iterations = ImGui.Int(self.fluid_simulator.iterations)
        borders = ImGui.Bool(self.fluid_simulator.has_borders)
        fluid_dissipation = ImGui.Float(self.fluid_simulator.dissipation)

        ImGui.text("Fluid simulation parameters")

        if ImGui.slider_float("Vorticity", vorticity, 0.0, 10.0):
            self.fluid_simulator.vorticity = vorticity.value

        if ImGui.slider_float("Viscosity", viscosity, 0.0, 10.0):
            self.fluid_simulator.viscosity = viscosity.value

        if ImGui.slider_float("Speed", speed, 1.0, 1000.0):
            self.fluid_simulator.speed = speed.value
            self.particle_area.speed = speed.value

        if ImGui.slider_float("V_Dissipation", fluid_dissipation, 0.500, 1.0):
            self.fluid_simulator.dissipation = fluid_dissipation.value

        if ImGui.slider_int("Iteration", iterations, 10, 100):
            self.fluid_simulator.iterations = iterations.value

        if ImGui.checkbox("Borders", borders):
            self.fluid_simulator.has_borders = borders.value

        ImGui.separator()

        particles_dissipation = ImGui.Float(self.particle_area.dissipation)

        ImGui.text("Particles area parameters")

        if ImGui.slider_float("P_Dissipation", particles_dissipation, 0.900, 1.0):
            self.particle_area.dissipation = particles_dissipation.value

        ImGui.separator()

        stop = ImGui.Bool(not self.fluid_simulator.simulate)

        if ImGui.checkbox("Stop simulation", stop):
            self.fluid_simulator.simulate = not stop.value
            self.particle_area.simulate = not stop.value

        ImGui.checkbox("Show velocity field", self.show_quiver_plot_overlay)
        ImGui.same_line()
        ImGui.push_item_width(200.0)
        ImGui.combo(
            "", self.quiver_plot_resolution, ["Very Small", "Small", "Medium", "Large"],
        )
        ImGui.pop_item_width()
        ImGui.end()


if __name__ == "__main__":
    demo = SimulationDemo(1280, 720, "Fluid Simulation Demo")
    demo.run()
