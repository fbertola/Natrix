from math import ceil

import imageio
import moderngl
import numpy as np
from moderngl import Context

from natrix.core.common.constants import TemplateConstants
from natrix.core.fluid_simulator import FluidSimulator
from natrix.core.utils.shaders_utils import read_shader_source, create_point_buffer


class ParticleArea:
    READ = 0
    WRITE = 1

    _width = 512
    _height = 512
    _particles_buffer = None

    _speed = 500.0
    _dissipation = 1.0

    def __init__(self, context: Context, width=512, height=512):
        self.context = context

        self._width = width
        self._height = height

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

    @property
    def particles(self):
        if not self._particles_buffer[self.READ]:
            raise RuntimeError('Particles buffer is empty')

        return np.frombuffer(self._particles_buffer[self.READ].read(), dtype=np.float32)

    def add_particles(self, position: tuple, radius: float, strength: float):
        self._add_particles_kernel["_Position"].value = position
        self._add_particles_kernel["_Value"].value = strength
        self._add_particles_kernel["_Radius"].value = radius
        self._add_particles_kernel.run(self._num_groups_x, self._num_groups_y, 1)
        self._flip_buffer()

    def update(self, time_delta: float):
        self._advect_particles_kernel['_Dissipation'].value = self.dissipation
        self._advect_particles_kernel['_ElapsedTime'].value = time_delta
        self._advect_particles_kernel['_Speed'].value = self.speed
        self._advect_particles_kernel.run(self._num_groups_x, self._num_groups_y, 1)
        self._flip_buffer()

    def _set_size(self):
        particle_size = (self._width, self._height)
        velocity_size = (self._width, self._height)
        group_size_x = TemplateConstants.NUM_THREADS.value
        group_size_y = TemplateConstants.NUM_THREADS.value

        self._num_cells = self._width * self._height
        self._num_groups_x = int(ceil(float(self._width) / float(group_size_x)))
        self._num_groups_y = int(ceil(float(self._height) / float(group_size_y)))

        self._add_particles_kernel["_ParticleSize"].value = particle_size
        self._advect_particles_kernel["_ParticleSize"].value = particle_size
        self._advect_particles_kernel["_VelocitySize"].value = velocity_size

    def _init_compute_kernels(self):
        self._particles_buffer[self.READ].bind_to_storage_buffer(
            TemplateConstants.PARTICLES_IN.value
        )
        self._particles_buffer[self.WRITE].bind_to_storage_buffer(
            TemplateConstants.PARTICLES_OUT.value
        )

    def _create_buffers(self):
        self._particles_buffer = [
            create_point_buffer(self.context, self._num_cells),
            create_point_buffer(self.context, self._num_cells),
        ]

    def _load_compute_kernels(self):
        self._add_particles_kernel = self.context.compute_shader(
            read_shader_source("shader.AddParticle.comp")
        )
        self._advect_particles_kernel = self.context.compute_shader(
            read_shader_source("shader.AdvectParticle.comp")
        )

    def _flip_buffer(self):
        tmp = self.READ
        self.READ = self.WRITE
        self.WRITE = tmp

        self._particles_buffer[self.READ].bind_to_storage_buffer(
            TemplateConstants.PARTICLES_IN.value
        )
        self._particles_buffer[self.WRITE].bind_to_storage_buffer(
            TemplateConstants.PARTICLES_OUT.value
        )

    def __del__(self):
        self._particles_buffer[0].release()
        self._particles_buffer[1].release()
