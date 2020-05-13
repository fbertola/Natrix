from ctypes import c_float
from math import ceil
from pathlib import Path

from bgfx import as_void_ptr, bgfx, ShaderType, load_shader

from natrix.core.common.constants import TemplateConstants
from natrix.core.fluid_simulator import FluidSimulator
from natrix.core.utils.shaders_utils import create_buffer

root_path = Path(__file__).parent / "shaders"


class SmoothParticlesArea:
    PARTICLES_IN = 0
    PARTICLES_OUT = 1

    _width = 512
    _height = 512
    _particles_buffer = None

    _speed = 500.0
    _dissipation = 1.0

    simulate = True

    def __init__(
        self,
        width: int,
        height: int,
        fluid_simulation: FluidSimulator,
        vertex_layout: bgfx.VertexLayout,
    ):
        self.fluid_simulation = fluid_simulation
        self.vertex_layout = vertex_layout

        self._width = width
        self._height = height

        self._create_uniforms()

        self._load_compute_kernels()
        self._set_size()
        self._create_buffers()
        self._init_compute_kernels()

    @property
    def speed(self):
        return self._speed

    @speed.setter
    def speed(self, value):
        if value > 0:
            self._speed = value
        else:
            raise ValueError("'Speed' should be greater than zero")

    @property
    def dissipation(self):
        return self._dissipation

    @dissipation.setter
    def dissipation(self, value):
        if value > 0:
            self._dissipation = value
        else:
            raise ValueError("'Dissipation' should be grater than zero")

    def add_particles(self, position: tuple, radius: float, strength: float):
        if self.simulate:
            self._init_compute_kernels()
            bgfx.set_uniform(
                self.position_uniform,
                as_void_ptr((c_float * 2)(position[0], position[1])),
            )
            bgfx.set_uniform(self.value_uniform, as_void_ptr((c_float * 1)(strength)))
            bgfx.set_uniform(self.radius_uniform, as_void_ptr((c_float * 1)(radius)))

            bgfx.dispatch(
                0, self._add_particles_kernel, self._num_groups_x, self._num_groups_x, 1
            )
            self._flip_buffer()

    def update(self, time_delta: float):
        self._init_compute_kernels()

        if self.simulate:
            bgfx.set_uniform(
                self.dissipation_uniform, as_void_ptr((c_float * 1)(self.dissipation))
            )
            bgfx.set_uniform(
                self.elapsed_time_uniform, as_void_ptr((c_float * 1)(time_delta))
            )
            bgfx.set_uniform(self.speed_uniform, as_void_ptr((c_float * 1)(self.speed)))

            bgfx.dispatch(
                0,
                self._advect_particles_kernel,
                self._num_groups_x,
                self._num_groups_y,
                1,
            )
            self._flip_buffer()

    def _set_size(self):
        self._particle_size = (self._width, self._height)
        self._velocity_size = (
            self.fluid_simulation.width,
            self.fluid_simulation.height,
        )
        group_size_x = TemplateConstants.NUM_THREADS.value
        group_size_y = TemplateConstants.NUM_THREADS.value

        self._num_cells = self._width * self._height
        self._num_groups_x = int(ceil(float(self._width) / float(group_size_x)))
        self._num_groups_y = int(ceil(float(self._height) / float(group_size_y)))

    def _create_uniforms(self):
        self.particle_size_uniform = bgfx.create_uniform(
            "_ParticleSize", bgfx.UniformType.VEC4
        )
        self.position_uniform = bgfx.create_uniform("_Position", bgfx.UniformType.VEC4)
        self.value_uniform = bgfx.create_uniform("_Value", bgfx.UniformType.VEC4)
        self.radius_uniform = bgfx.create_uniform("_Radius", bgfx.UniformType.VEC4)
        self.dissipation_uniform = bgfx.create_uniform(
            "_Dissipation", bgfx.UniformType.VEC4
        )
        self.elapsed_time_uniform = bgfx.create_uniform(
            "_ElapsedTime", bgfx.UniformType.VEC4
        )
        self.speed_uniform = bgfx.create_uniform("_Speed", bgfx.UniformType.VEC4)
        self.velocity_size_uniform = bgfx.create_uniform(
            "_VelocitySize", bgfx.UniformType.VEC4
        )

    def _init_compute_kernels(self):
        bgfx.set_uniform(
            self.particle_size_uniform,
            as_void_ptr((c_float * 2)(self._particle_size[0], self._particle_size[1])),
        )
        bgfx.set_uniform(
            self.velocity_size_uniform,
            as_void_ptr((c_float * 2)(self._velocity_size[0], self._velocity_size[1])),
        )

        bgfx.set_buffer(
            TemplateConstants.PARTICLES_IN.value,
            self._particles_buffer[self.PARTICLES_IN],
            bgfx.Access.WRITE,
        )
        bgfx.set_buffer(
            TemplateConstants.PARTICLES_OUT.value,
            self._particles_buffer[self.PARTICLES_OUT],
            bgfx.Access.WRITE,
        )

    def _create_buffers(self):
        self._particles_buffer = [
            create_buffer(self._num_cells, 1, self.vertex_layout),
            create_buffer(self._num_cells, 1, self.vertex_layout),
        ]

    def _load_compute_kernels(self):
        self._add_particles_kernel = bgfx.create_program(
            load_shader(
                "shader.AddParticle.comp", ShaderType.COMPUTE, root_path=root_path
            ),
            True,
        )
        self._advect_particles_kernel = bgfx.create_program(
            load_shader(
                "shader.AdvectParticle.comp", ShaderType.COMPUTE, root_path=root_path
            ),
            True,
        )

    def _flip_buffer(self):
        tmp = self.PARTICLES_IN
        self.PARTICLES_IN = self.PARTICLES_OUT
        self.PARTICLES_OUT = tmp

        bgfx.set_buffer(
            TemplateConstants.PARTICLES_IN.value,
            self._particles_buffer[self.PARTICLES_IN],
            bgfx.Access.READ,
        )
        bgfx.set_buffer(
            TemplateConstants.PARTICLES_OUT.value,
            self._particles_buffer[self.PARTICLES_OUT],
            bgfx.Access.WRITE,
        )

    def destroy(self):
        # Destroy uniforms
        bgfx.destroy(self.particle_size_uniform)
        bgfx.destroy(self.position_uniform)
        bgfx.destroy(self.value_uniform)
        bgfx.destroy(self.radius_uniform)
        bgfx.destroy(self.dissipation_uniform)
        bgfx.destroy(self.elapsed_time_uniform)
        bgfx.destroy(self.speed_uniform)
        bgfx.destroy(self.velocity_size_uniform)

        # Destroy buffers
        bgfx.destroy(self._particles_buffer[0])
        bgfx.destroy(self._particles_buffer[1])

        # Destroy compute shaders
        bgfx.destroy(self._add_particles_kernel)
        bgfx.destroy(self._advect_particles_kernel)
