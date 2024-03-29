import ctypes
from ctypes import c_float, sizeof, Structure
from pathlib import Path

import glfw
import numpy as np
from pybgfx import (
    bgfx,
    ImGui,
)
from pybgfx.utils.imgui_utils import ImGuiExtra
from pybgfx.constants import (
    BGFX_CLEAR_COLOR,
    BGFX_CLEAR_DEPTH,
    BGFX_RESET_VSYNC,
    BGFX_DEBUG_TEXT,
    BGFX_STATE_WRITE_RGB,
    BGFX_STATE_WRITE_A,
    BGFX_STATE_DEFAULT,
    BGFX_TEXTURE_COMPUTE_WRITE,
    BGFX_RESET_HIDPI,
    BGFX_STATE_BLEND_ALPHA,
)
from pybgfx.utils import as_void_ptr
from pybgfx.utils.shaders_utils import ShaderType, load_shader
from loguru import logger

from demo.example_window import ExampleWindow
from demo.smooth_particles_area import SmoothParticlesArea
# from demo.utils.imgui_utils import show_properties_dialog
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

        self.particles_strength = 0.04
        self.particles_diameter = 250.0

        self.show_quiver_plot_overlay = ctypes.c_bool(False)
        self.quiver_plot_resolution = ctypes.c_int(2)
        self.quiver_plot_resolutions = (8.0, 16.0, 32.0, 64.0)

    def init(self, platform_data):
        bgfx.renderFrame()
        bgfx.setPlatformData(platform_data)
        bgfx.init(self.init_conf)
        bgfx.reset(
            self.fb_width,
            self.fb_height,
            BGFX_RESET_VSYNC | BGFX_RESET_HIDPI,
            self.init_conf.resolution.format,
        )

        bgfx.setDebug(BGFX_DEBUG_TEXT)
        # bgfx.setViewClear(0, BGFX_CLEAR_COLOR | BGFX_CLEAR_DEPTH, 0xFFFFFFFF, 1.0, 0)
        bgfx.setViewClear(0, BGFX_CLEAR_COLOR | BGFX_CLEAR_DEPTH, 0x1a0427FF, 1.0, 0)

        self.vertex_layout = bgfx.VertexLayout()
        self.vertex_layout.begin().add(
            bgfx.Attrib.Position, 3, bgfx.AttribType.Float
        ).add(bgfx.Attrib.TexCoord0, 3, bgfx.AttribType.Float).end()

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
        self.vertex_buffer = bgfx.createVertexBuffer(vb_memory, self.vertex_layout)

        ib_memory = bgfx.copy(as_void_ptr(cube_indices), cube_indices.nbytes)
        self.index_buffer = bgfx.createIndexBuffer(ib_memory)

        self.output_texture = bgfx.createTexture2D(
            self.fb_width,
            self.fb_height,
            False,
            1,
            bgfx.TextureFormat.RGBA8,
            BGFX_TEXTURE_COMPUTE_WRITE,
        )

        self.texture_uniform = bgfx.createUniform(
            "InputTexture", bgfx.UniformType.Sampler
        )
        self.window_size_uniform = bgfx.createUniform(
            "WindowSize", bgfx.UniformType.Vec4
        )
        self.velocity_size_uniform = bgfx.createUniform(
            "VelocitySize", bgfx.UniformType.Vec4
        )
        self.arrow_tile_size_uniform = bgfx.createUniform(
            "ArrowTileSize", bgfx.UniformType.Vec4
        )

        # Create program from shaders.
        self.main_program = bgfx.createProgram(
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
        self.quiver_program = bgfx.createProgram(
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
        self.cs_program = bgfx.createProgram(
            load_shader(
                "demo.ComputeShader.comp", ShaderType.COMPUTE, root_path=root_path
            ),
            True,
        )

        ImGuiExtra.create(36.0)

    def shutdown(self):
        self.fluid_simulator.destroy()
        self.particle_area.destroy()

        ImGuiExtra.destroy()

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

        ImGuiExtra.begin_frame(
            int(mouse_x), int(mouse_y), buttons_states, 0, self.fb_width, self.fb_height
        )
        show_properties_dialog(self.fb_width, self.fb_height, self.hidpi)
        self._create_imgui_config_dialog()

        ImGuiExtra.end_frame()

        at = (c_float * 3)(*[0.0, 0.0, 0.0])
        eye = (c_float * 3)(*[0.0, 0.0, 10.0])
        up = (c_float * 3)(*[0.0, 1.0, 0.0])

        view = look_at(eye, at, up)
        projection = proj(11.4, 1, 0.1, 100.0)

        bgfx.setViewRect(0, 0, 0, self.fb_width, self.fb_height)

        bgfx.setViewTransform(0, as_void_ptr(view), as_void_ptr(projection))

        bgfx.setVertexBuffer(0, self.vertex_buffer, 0, 4)
        bgfx.setIndexBuffer(self.index_buffer, 0, cube_indices.size)

        bgfx.setState(BGFX_STATE_DEFAULT)
        bgfx.setImage(0, self.output_texture, 0, bgfx.Access.Write)

        self.fluid_simulator.add_circle_obstacle((0.5, 0.5), 40.0)

        self.fluid_simulator.update(dt)
        self.particle_area.update(dt)

        if glfw.get_mouse_button(self.window, glfw.MOUSE_BUTTON_LEFT) == glfw.PRESS\
                and not ImGui.GetIO().WantCaptureMouse:
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
            self.particle_area.add_particles((n_mouse_x, n_mouse_y), self.particles_diameter, self.particles_strength)

        self.old_mouse_x = mouse_x
        self.old_mouse_y = mouse_y

        bgfx.setState(
            0 | BGFX_STATE_WRITE_RGB | BGFX_STATE_WRITE_A | BGFX_STATE_BLEND_ALPHA
        )

        bgfx.dispatch(0, self.cs_program, self.fb_width // 16, self.fb_height // 16)
        bgfx.setTexture(0, self.texture_uniform, self.output_texture)
        bgfx.submit(0, self.main_program, 0, False)

        if self.show_quiver_plot_overlay.value:
            bgfx.setBuffer(
                1, self.fluid_simulator.get_velocity_buffer(), bgfx.Access.Read
            )
            bgfx.setUniform(
                self.window_size_uniform,
                as_void_ptr((c_float * 2)(self.fb_width, self.fb_height)),
            )
            bgfx.setUniform(
                self.velocity_size_uniform,
                as_void_ptr(
                    (c_float * 2)(
                        self.fluid_simulator.width, self.fluid_simulator.height
                    )
                ),
            )
            bgfx.setUniform(
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

        ImGui.SetNextWindowPos(
            ImGui.ImVec2(
                self.fb_width - self.fb_width / 4.1 - 20.0 * res_multiplier,
                40.0 * res_multiplier,
            ),
            ImGui.ImGuiCond_FirstUseEver,
        )
        ImGui.SetNextWindowSize(
            ImGui.ImVec2(self.fb_width / 4.1, self.fb_height / 2.1),
            ImGui.ImGuiCond_FirstUseEver,
        )

        ImGui.Begin("\uf013 Settings")

        vorticity = ctypes.c_float(self.fluid_simulator.vorticity)
        viscosity = ctypes.c_float(self.fluid_simulator.viscosity)
        speed = ctypes.c_float(self.fluid_simulator.speed)
        iterations = ctypes.c_int(self.fluid_simulator.iterations)
        borders = ctypes.c_bool(self.fluid_simulator.has_borders)
        fluid_dissipation = ctypes.c_float(self.fluid_simulator.dissipation)
        part_strength = ctypes.c_float(self.particles_strength)
        part_diameter = ctypes.c_float(self.particles_diameter)

        ImGui.Text("Fluid simulation parameters")

        if ImGui.SliderFloat("Vorticity", vorticity, 0.0, 10.0):
            self.fluid_simulator.vorticity = vorticity.value

        if ImGui.SliderFloat("Viscosity", viscosity, 0.0, 10.0):
            self.fluid_simulator.viscosity = viscosity.value

        if ImGui.SliderFloat("Speed", speed, 1.0, 1000.0):
            self.fluid_simulator.speed = speed.value
            self.particle_area.speed = speed.value

        if ImGui.SliderFloat("V_Dissipation", fluid_dissipation, 0.500, 1.0):
            self.fluid_simulator.dissipation = fluid_dissipation.value

        if ImGui.SliderInt("Iteration", iterations, 10, 100):
            self.fluid_simulator.iterations = iterations.value

        if ImGui.Checkbox("Borders", borders):
            self.fluid_simulator.has_borders = borders.value

        ImGui.Separator()

        particles_dissipation = ctypes.c_float(self.particle_area.dissipation)

        ImGui.Text("Particles area parameters")

        if ImGui.SliderFloat("P_Dissipation", particles_dissipation, 0.900, 1.0):
            self.particle_area.dissipation = particles_dissipation.value

        if ImGui.SliderFloat("Strength", part_strength, 0.0, 1.0):
            self.particles_strength = part_strength.value

        if ImGui.SliderFloat("Diameter", part_diameter, 100.0, 1000.0):
            self.particles_diameter = part_diameter.value

        ImGui.Separator()

        stop = ctypes.c_bool(not self.fluid_simulator.simulate)

        if ImGui.Checkbox("Stop simulation", stop):
            self.fluid_simulator.simulate = not stop.value
            self.particle_area.simulate = not stop.value

        ImGui.Checkbox("Show velocity field", self.show_quiver_plot_overlay)
        ImGui.SameLine()
        ImGui.PushItemWidth(200.0)
        ImGui.Combo(
            "", self.quiver_plot_resolution, ["Very Small", "Small", "Medium", "Large"], 4,
        )
        ImGui.PopItemWidth()
        ImGui.End()


if __name__ == "__main__":
    demo = SimulationDemo(1280, 720, "Fluid Simulation Demo")
    demo.run()
