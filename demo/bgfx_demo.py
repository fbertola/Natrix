from ctypes import Structure, c_float, sizeof
from pathlib import Path

import numpy as np
from bgfx import bgfx, ImGuiExtra, BGFX_CLEAR_COLOR, BGFX_CLEAR_DEPTH, BGFX_RESET_VSYNC, BGFX_DEBUG_TEXT, as_void_ptr, \
    ShaderType, load_shader, BGFX_STATE_WRITE_RGB, BGFX_STATE_WRITE_A, BGFX_STATE_DEFAULT, BGFX_TEXTURE_COMPUTE_WRITE, \
    BGFX_STATE_BLEND_ALPHA, BGFX_TEXTURE_RT

from natrix.core.example_window import ExampleWindow
from natrix.core.fluid_simulator import FluidSimulator
from natrix.core.particle_area import ParticleArea
from natrix.core.utils.imgui_utils import show_example_dialog
from natrix.core.utils.matrix_utils import look_at, proj, rotate_xy
from natrix.core.utils.shaders_utils import create_buffer

import random


class PosColorVertex(Structure):
    _fields_ = [("m_x", c_float),
                ("m_y", c_float),
                ("m_z", c_float),
                ("m_u", c_float),
                ("m_v", c_float),
                ("m_w", c_float), ]


cube_vertices = (PosColorVertex * 4)(
    PosColorVertex(-1.0, 1.0, 0.0, -1.0, 1.0, 1.0),
    PosColorVertex(1.0, 1.0, 0.0, 1.0, 1.0, 1.0),
    PosColorVertex(-1.0, -1.0, 0.0, -1.0, -1.0, 1.0),
    PosColorVertex(1.0, -1.0, 0.0, 1.0, -1.0, 1.0),
)

cube_indices = np.array([
    0, 1, 2,
    1, 2, 3,
], dtype=np.uint16)

root_path = Path(__file__).parent / "shaders"


class Demo(ExampleWindow):
    def __init__(self, width, height, title):
        super().__init__(width, height, title)

        self.init_conf = bgfx.Init()
        self.init_conf.debug = True
        self.init_conf.type = bgfx.RendererType.Metal
        self.init_conf.resolution.width = self.width
        self.init_conf.resolution.height = self.height
        self.init_conf.resolution.reset = BGFX_RESET_VSYNC

    def init(self, platform_data):
        self.init_conf.platformData = platform_data

        bgfx.init(self.init_conf)
        bgfx.reset(self.width, self.height, BGFX_RESET_VSYNC, self.init_conf.resolution.format)

        bgfx.setDebug(BGFX_DEBUG_TEXT)
        bgfx.setViewClear(0, BGFX_CLEAR_COLOR | BGFX_CLEAR_DEPTH, 0x443355FF, 1.0, 0)

        self.vertex_layout = bgfx.VertexLayout()
        self.vertex_layout.begin() \
            .add(bgfx.Attrib.Position, 3, bgfx.AttribType.Float) \
            .add(bgfx.Attrib.TexCoord0, 3, bgfx.AttribType.Float) \
            .end()

        self.fluid_simulator = FluidSimulator(512, 512, self.vertex_layout)
        self.fluid_simulator.vorticity = 5.0
        self.fluid_simulator.viscosity = 0.000000000001

        # Create static vertex buffer
        vb_memory = bgfx.copy(as_void_ptr(cube_vertices),
                              sizeof(PosColorVertex) * 4)
        self.vertex_buffer = bgfx.createVertexBuffer(vb_memory, self.vertex_layout)

        ib_memory = bgfx.copy(as_void_ptr(cube_indices), cube_indices.nbytes)
        self.index_buffer = bgfx.createIndexBuffer(ib_memory)

        self.output_texture = bgfx.createTexture2D(
            512
            , 512
            , False
            , 1
            , bgfx.TextureFormat.RGBA32F
            , BGFX_TEXTURE_COMPUTE_WRITE
        )

        self.texture_uniform = bgfx.createUniform("InputTexture", bgfx.UniformType.Sampler)

        # Create program from shaders.
        self.main_program = bgfx.createProgram(
            load_shader("demo.VertexShader.vert", ShaderType.VERTEX, root_path=root_path),
            load_shader("demo.FragmentShader.frag", ShaderType.FRAGMENT, root_path=root_path),
            True)
        self.cs_program = bgfx.createProgram(load_shader('demo.ComputeShader.comp', ShaderType.COMPUTE, root_path=root_path), True)

        ImGuiExtra.imguiCreate()

    def shutdown(self):
        self.fluid_simulator.destroy()

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
            int(mouse_x), int(mouse_y), buttons_states, 0, self.width, self.height
        )

        show_example_dialog()

        ImGuiExtra.imguiEndFrame()

        vel_x = random.uniform(-0.08, 0.08)
        vel_y = random.uniform(-0.01, 0.2)

        at = (c_float * 3)(*[0.0, 0.0, 0.0])
        eye = (c_float * 3)(*[0.0, 0.0, 10.0])
        up = (c_float * 3)(*[0.0, 1.0, 0.0])

        view = look_at(eye, at, up)
        projection = proj(15.0, self.width / self.height, 0.1, 100.0)

        bgfx.setViewRect(
            0, 0, 0, self.width, self.height
        )
        bgfx.touch(0)

        bgfx.setViewTransform(0, as_void_ptr(view), as_void_ptr(projection))

        # Set vertex and index buffer.
        bgfx.setVertexBuffer(0, self.vertex_buffer, 0, 4)
        bgfx.setIndexBuffer(self.index_buffer, 0, cube_indices.size)

        bgfx.setState(BGFX_STATE_DEFAULT)
        bgfx.setImage(0, self.output_texture, 0, bgfx.Access.Write)

        self.fluid_simulator.add_circle_obstacle((0.5, 0.5), 30)
        self.fluid_simulator.add_velocity((0.5, 0.1), (vel_x, vel_y), 44.0)
        self.fluid_simulator.update(dt)

        bgfx.dispatch(0, self.cs_program, 512 // 16, 512 // 16)
        bgfx.setTexture(0, self.texture_uniform, self.output_texture)
        bgfx.setState(BGFX_STATE_WRITE_RGB | BGFX_STATE_WRITE_A )
        bgfx.submit(0, self.main_program, 0, False)

        bgfx.frame()


if __name__ == "__main__":
    demo = Demo(1280, 720, "demo/bgfx_demo")
    demo.run()
