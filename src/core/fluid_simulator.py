from math import ceil

import moderngl
import numpy as np
from moderngl import Context

from src.core.common.constants import TemplateConstants
from src.core.utils.shaders_utils import read_shader_source
import imageio


class FluidSimulator:
    VELOCITY_READ = 0
    VELOCITY_WRITE = 1

    PRESSURE_READ = 0
    PRESSURE_WRITE = 1

    _num_cells = 0
    _num_groups_x = 0
    _num_groups_y = 0
    _width = 512
    _height = 512

    # Public properties
    speed = 500.0
    iterations = 50
    velocity_dissipation = 1.0
    vorticity = 0.0
    viscosity = 0.1
    resolution = 512
    simulate = True

    def __init__(self, context: Context, width: int, height: int):
        self.context = context

        self._load_compute_kernels()
        self._set_size(width, height)
        self._create_buffers()
        self._init_compute_kernels()

    def add_velocity(self, position: tuple, velocity: tuple, radius: float):
        self._add_velocity_kernel["_Position"].value = position
        self._add_velocity_kernel["_Value"].value = velocity
        self._add_velocity_kernel["_Radius"].value = radius

        self._add_velocity_kernel.run(self._num_groups_x, self._num_groups_y, 1)
        self._flip_velocity_buffer()

    def update(self, time_delta: float):
        if self.simulate:
            self._update_params(time_delta)

            # Init boundaries
            self._init_boundaries_kernel.run(self._num_groups_x, self._num_groups_y, 1)

            # Advect
            self._advect_velocity_kernel.run(self._num_groups_x, self._num_groups_y, 1)
            self._flip_velocity_buffer()

            # Vorticity confinement 1 - Calculate vorticity
            self._calc_vorticity_kernel.run(self._num_groups_x, self._num_groups_y, 1)

            # Vorticity confinement 2 - Apply vorticity force
            self._apply_vorticity_kernel.run(self._num_groups_x, self._num_groups_y, 1)
            self._flip_velocity_buffer()

            # Viscosity
            if self.viscosity > 0.0:
                for _ in range(self.iterations):
                    self._viscosity_kernel.run(
                        self._num_groups_x, self._num_groups_y, 1
                    )
                    self._flip_velocity_buffer()

            # Divergence
            self._divergence_kernel.run(self._num_groups_x, self._num_groups_y, 1)

            # Clear pressure
            self._pressure_buffer[self.PRESSURE_READ].bind_to_storage_buffer(
                TemplateConstants.GENERIC.value
            )
            self._clear_buffer_kernel.run(self._num_groups_x, self._num_groups_y, 1)
            self._pressure_buffer[self.PRESSURE_READ].bind_to_storage_buffer(
                TemplateConstants.PRESSURE_IN.value
            )

            # Poisson
            for _ in range(self.iterations):
                self._poisson_kernel.run(self._num_groups_x, self._num_groups_y, 1)
                self._flip_pressure_buffer()

            # Subtract gradient
            self._subtract_gradient_kernel.run(
                self._num_groups_x, self._num_groups_y, 1
            )
            self._flip_velocity_buffer()

            # Clear obstacles
            self._obstacles_buffer.bind_to_storage_buffer(
                TemplateConstants.GENERIC.value
            )
            self._clear_buffer_kernel.run(self._num_groups_x, self._num_groups_y, 1)
            self._obstacles_buffer.bind_to_storage_buffer(
                TemplateConstants.OBSTACLES.value
            )

    def _set_size(self, width: int, height: int):
        group_size_x = TemplateConstants.NUM_THREADS.value
        group_size_y = TemplateConstants.NUM_THREADS.value

        self._width = width
        self._height = height
        self._num_cells = width * height
        self._num_groups_x = int(ceil(float(width) / float(group_size_x)))
        self._num_groups_y = int(ceil(float(height) / float(group_size_y)))

    def _update_params(self, time_delta: float):

        self._advect_velocity_kernel["_ElapsedTime"].value = time_delta
        self._advect_velocity_kernel["_Speed"].value = self.speed
        self._advect_velocity_kernel["_Dissipation"].value = self.velocity_dissipation

        self._apply_vorticity_kernel["_ElapsedTime"].value = time_delta
        self._apply_vorticity_kernel["_VorticityScale"].value = self.vorticity

        centre_factor = 1.0 / self.viscosity
        stencil_factor = 1.0 / (4.0 + centre_factor)

        self._viscosity_kernel["_Alpha"].value = centre_factor
        self._viscosity_kernel["_rBeta"].value = stencil_factor

    def _init_compute_kernels(self):
        size = (self._width, self._height)

        self._add_velocity_kernel["_Size"].value = size
        self._init_boundaries_kernel["_Size"].value = size
        self._advect_velocity_kernel["_Size"].value = size
        self._divergence_kernel["_Size"].value = size
        self._poisson_kernel["_Size"].value = size
        self._subtract_gradient_kernel["_Size"].value = size
        self._calc_vorticity_kernel["_Size"].value = size
        self._apply_vorticity_kernel["_Size"].value = size
        self._add_circle_obstacle_kernel["_Size"].value = size
        self._add_triangle_obstacle_kernel["_Size"].value = size
        self._clear_buffer_kernel["_Size"].value = size
        self._viscosity_kernel["_Size"].value = size

        self._velocity_buffer[self.VELOCITY_READ].bind_to_storage_buffer(
            TemplateConstants.VELOCITY_IN.value
        )
        self._velocity_buffer[self.VELOCITY_WRITE].bind_to_storage_buffer(
            TemplateConstants.VELOCITY_OUT.value
        )
        self._pressure_buffer[self.PRESSURE_READ].bind_to_storage_buffer(
            TemplateConstants.PRESSURE_IN.value
        )
        self._pressure_buffer[self.PRESSURE_WRITE].bind_to_storage_buffer(
            TemplateConstants.PRESSURE_OUT.value
        )
        self._divergence_buffer.bind_to_storage_buffer(
            TemplateConstants.DIVERGENCE.value
        )
        self._vorticity_buffer.bind_to_storage_buffer(TemplateConstants.VORTICITY.value)
        self._obstacles_buffer.bind_to_storage_buffer(TemplateConstants.OBSTACLES.value)

    def _create_buffers(self):
        self._velocity_buffer = [
            self._create_vector_buffer(),
            self._create_vector_buffer(),
        ]
        self._pressure_buffer = [
            self._create_point_buffer(),
            self._create_point_buffer(),
        ]
        self._divergence_buffer = self._create_point_buffer()
        self._vorticity_buffer = self._create_point_buffer()
        self._obstacles_buffer = self._create_vector_buffer()

    def _load_compute_kernels(self):
        self._add_velocity_kernel = self.context.compute_shader(
            read_shader_source("shader.AddVelocity.comp")
        )
        self._init_boundaries_kernel = self.context.compute_shader(
            read_shader_source("shader.InitBoundaries.comp")
        )
        self._advect_velocity_kernel = self.context.compute_shader(
            read_shader_source("shader.AdvectVelocity.comp")
        )
        self._divergence_kernel = self.context.compute_shader(
            read_shader_source("shader.Divergence.comp")
        )
        self._poisson_kernel = self.context.compute_shader(
            read_shader_source("shader.Poisson.comp")
        )
        self._subtract_gradient_kernel = self.context.compute_shader(
            read_shader_source("shader.SubtractGradient.comp")
        )
        self._calc_vorticity_kernel = self.context.compute_shader(
            read_shader_source("shader.CalcVorticity.comp")
        )
        self._apply_vorticity_kernel = self.context.compute_shader(
            read_shader_source("shader.ApplyVorticity.comp")
        )
        self._add_circle_obstacle_kernel = self.context.compute_shader(
            read_shader_source("shader.AddCircleObstacle.comp")
        )
        self._add_triangle_obstacle_kernel = self.context.compute_shader(
            read_shader_source("shader.AddTriangleObstacle.comp")
        )
        self._clear_buffer_kernel = self.context.compute_shader(
            read_shader_source("shader.ClearBuffer.comp")
        )
        self._viscosity_kernel = self.context.compute_shader(
            read_shader_source("shader.Viscosity.comp")
        )

    def _create_point_buffer(self, default_value=0.0):
        data = [default_value for _ in range(self._num_cells)]
        data_bytes = np.array(data).astype(np.float32).tobytes()

        return self.context.buffer(data_bytes)

    def _create_vector_buffer(self, default_value_x=0.0, default_value_y=0.0):
        data = [[default_value_x, default_value_y] for _ in range(self._num_cells)]
        data_bytes = np.array(data).astype(np.float32).tobytes()

        return self.context.buffer(data_bytes)

    def _flip_velocity_buffer(self):
        tmp = self.VELOCITY_READ
        self.VELOCITY_READ = self.VELOCITY_WRITE
        self.VELOCITY_WRITE = tmp

        self._velocity_buffer[self.VELOCITY_READ].bind_to_storage_buffer(
            TemplateConstants.VELOCITY_IN.value
        )
        self._velocity_buffer[self.VELOCITY_WRITE].bind_to_storage_buffer(
            TemplateConstants.VELOCITY_OUT.value
        )

    def _flip_pressure_buffer(self):
        tmp = self.PRESSURE_READ
        self.PRESSURE_READ = self.PRESSURE_WRITE
        self.PRESSURE_WRITE = tmp

        self._pressure_buffer[self.PRESSURE_READ].bind_to_storage_buffer(
            TemplateConstants.PRESSURE_IN.value
        )
        self._pressure_buffer[self.PRESSURE_WRITE].bind_to_storage_buffer(
            TemplateConstants.PRESSURE_OUT.value
        )

    def __del__(self):
        self._velocity_buffer[0].release()
        self._velocity_buffer[1].release()
        self._pressure_buffer[0].release()
        self._pressure_buffer[1].release()
        self._divergence_buffer.release()
        self._vorticity_buffer.release()
        self._obstacles_buffer.release()


if __name__ == "__main__":
    simulator = FluidSimulator(moderngl.create_standalone_context(), 8, 8)

    simulator.add_velocity((0.5, 0.5), (0.5, -0.5), 1.0)

    for i in range(1):
        simulator.update(0.003)
        output = np.frombuffer(simulator._velocity_buffer[simulator.VELOCITY_READ].read(), dtype=np.float32)
        output = output.reshape((8, 8, 2))
        print(output)
