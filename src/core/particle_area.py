from math import ceil

import imageio
import moderngl
import numpy as np
from moderngl import Context

from src.core.common.constants import TemplateConstants
from src.core.fluid_simulator import FluidSimulator
from src.core.utils.shaders_utils import read_shader_source, create_point_buffer


class ParticleArea:
    READ = 0
    WRITE = 1

    _width = 512
    _height = 512
    _particles_buffer = None # FIXME: accessors

    speed = 500.0
    dissipation = 1.0

    def __init__(self, context: Context, width=512, height=512, resolution=128):
        self.context = context

        self._width = width
        self._height = height
        self._resolution = resolution

        self._load_compute_kernels()
        self._set_size()
        self._create_buffers()
        self._init_compute_kernels()

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
        velocity_size = (self._width, self._height) # FIXME: this should be configurable
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


if __name__ == "__main__":
    imgs = []
    context = moderngl.create_standalone_context()
    fluid_simulator = FluidSimulator(context, 512, 512)
    particle_area = ParticleArea(context, 512, 512)

    particle_area.add_particles((0.5, 0.5), 30.0, 1)

    for i in range(60*10):
        #particle_area.add_particles((0.5, 0.5), 30.0, 1)
        fluid_simulator.add_velocity((0.5, 0.5), (1.0, -1.0), 30.0)

        fluid_simulator.update(0.001)
        particle_area.update(0.001)

        output = np.frombuffer(particle_area._particles_buffer[particle_area.READ].read(), dtype=np.float32)
        output = output.reshape((512, 512, 1))
        output = np.multiply(output, 255).astype(np.uint8)
        #print(output)
        imgs.append(output)

    # if you don't want to use imageio, remove this line
    imageio.mimwrite("./debug.gif", imgs, "GIF")
