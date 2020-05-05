import random
from ctypes import Structure, c_float, sizeof
from pathlib import Path

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
)

from demo.example_window import ExampleWindow
from natrix.core.fluid_simulator import FluidSimulator
from demo.smooth_particles_area import SmoothParticlesArea
from demo.utils.imgui_utils import show_properties_dialog
from demo.utils.matrix_utils import look_at, proj


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

    def init(self, platform_data):
        self.init_conf.platformData = platform_data
        bgfx.renderFrame()
        bgfx.init(self.init_conf)
        bgfx.reset(
            self.fb_width,
            self.fb_height,
            BGFX_RESET_VSYNC | BGFX_RESET_HIDPI,
            self.init_conf.resolution.format,
        )

        bgfx.setDebug(BGFX_DEBUG_TEXT)
        bgfx.setViewClear(0, BGFX_CLEAR_COLOR | BGFX_CLEAR_DEPTH, 0x443355FF, 1.0, 0)

        self.vertex_layout = bgfx.VertexLayout()
        self.vertex_layout.begin().add(
            bgfx.Attrib.Position, 3, bgfx.AttribType.Float
        ).add(bgfx.Attrib.TexCoord0, 3, bgfx.AttribType.Float).end()

        self.fluid_simulator = FluidSimulator(
            self.width // 2, self.height // 2, self.vertex_layout
        )
        self.fluid_simulator.vorticity = 1.0
        self.fluid_simulator.viscosity = 0.0
        self.fluid_simulator.iterations = 100

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

        # Create program from shaders.
        self.main_program = bgfx.createProgram(
            load_shader(
                "demo.VertexShader.vert", ShaderType.VERTEX, root_path=root_path
            ),
            load_shader(
                "demo.FragmentShader.frag", ShaderType.FRAGMENT, root_path=root_path
            ),
            True,
        )
        self.cs_program = bgfx.createProgram(
            load_shader(
                "demo.ComputeShader.comp", ShaderType.COMPUTE, root_path=root_path
            ),
            True,
        )

        ImGuiExtra.imguiCreate(36.0)

    def shutdown(self):
        self.fluid_simulator.destroy()
        self.particle_area.destroy()

        ImGuiExtra.imguiDestroy()

        bgfx.destroy(self.index_buffer)
        bgfx.destroy(self.vertex_buffer)
        bgfx.destroy(self.output_texture)
        bgfx.destroy(self.cs_program)
        bgfx.destroy(self.texture_uniform)
        bgfx.destroy(self.main_program)

        bgfx.shutdown()

    def update(self, dt):
        mouse_x, mouse_y, buttons_states = self.get_mouse_state()
        ImGuiExtra.imguiBeginFrame(
            int(mouse_x), int(mouse_y), buttons_states, 0, self.fb_width, self.fb_height
        )
        show_properties_dialog(self.fluid_simulator, self.particle_area, self.hidpi)

        ImGuiExtra.imguiEndFrame()

        vel_y = random.uniform(-0.08, 0.08)
        vel_x = random.uniform(-0.01, 0.1)
        strength = random.uniform(0.01, 0.09)

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

        self.fluid_simulator.add_velocity((0.23, 0.5), (vel_x, vel_y), 34.0)
        self.fluid_simulator.add_circle_obstacle((0.5, 0.7), 30.0)
        self.fluid_simulator.add_triangle_obstacle(
            (0.65, 0.5), (0.42, 0.5), (0.42, 0.39)
        )
        self.fluid_simulator.add_triangle_obstacle(
            (0.65, 0.5), (0.65, 0.39), (0.42, 0.39)
        )
        self.fluid_simulator.update(dt)

        self.particle_area.add_particles((0.2, 0.5), 220.0, strength)
        self.particle_area.update(dt)

        bgfx.dispatch(0, self.cs_program, self.fb_width // 16, self.fb_height // 16)
        bgfx.setTexture(0, self.texture_uniform, self.output_texture)
        bgfx.setState(BGFX_STATE_WRITE_RGB | BGFX_STATE_WRITE_A)
        bgfx.submit(0, self.main_program, 0, False)
        bgfx.frame()

    def resize(self):
        bgfx.reset(
            self.fb_width,
            self.fb_height,
            BGFX_RESET_VSYNC | BGFX_RESET_HIDPI,
            self.init_conf.resolution.format,
        )


if __name__ == "__main__":
    demo = SimulationDemo(1280, 720, "Fluid Simulation Demo")
    demo.run()
