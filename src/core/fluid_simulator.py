import moderngl
import numpy as np
from moderngl import Context
from math import ceil
from src.core.utils.shaders_utils import read_shader_source


class FluidSimulator:
    VELOCITY_READ = 0
    VELOCITY_WRITE = 1

    PRESSURE_READ = 0
    PRESSURE_WRITE = 1

    # Compute Shader Kernel Ids
    _add_velocity_kernel = 0
    _init_boundaries_kernel = 0
    _advect_velocity_kernel = 0
    _divergence_kernel = 0
    _poisson_kernel = 0
    _subtract_gradient_kernel = 0
    _calc_vorticity_kernel = 0
    _apply_vorticity_kernel = 0
    _viscosityKernel = 0
    _add_circle_obstacle_kernel = 0
    _add_triangle_obstacle_kernel = 0
    _clear_buffer_kernel = 0

    _velocity_buffer = None
    _divergence_buffer = None
    _pressure_buffer = None
    _vorticity_buffer = None
    _obstacles_buffer = None

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
    viscosity = 0.0
    resolution = 512

    def __init__(self, context: Context):
        self.context = context

        self._add_velocity_kernel = context.compute_shader(read_shader_source('shader.AddVelocity.comp'))
        self._init_boundaries_kernel = context.compute_shader(read_shader_source('shader.InitBoundaries.comp'))
        self._advect_velocity_kernel = context.compute_shader(read_shader_source('shader.AdvectVelocity.comp'))
        self._divergence_kernel = context.compute_shader(read_shader_source('shader.Divergence.comp'))
        self._poisson_kernel = context.compute_shader(read_shader_source('shader.Poisson.comp'))
        self._subtract_gradient_kernel = context.compute_shader(read_shader_source('shader.SubtractGradient.comp'))
        self._calc_vorticity_kernel = context.compute_shader(read_shader_source('shader.CalcVorticity.comp'))
        self._apply_vorticity_kernel = context.compute_shader(read_shader_source('shader.ApplyVorticity.comp'))
        self._add_circle_obstacle_kernel = context.compute_shader(read_shader_source('shader.AddCircleObstacle.comp'))
        self._add_triangle_obstacle_kernel = context.compute_shader(
            read_shader_source('shader.AddTriangleObstacle.comp'))
        self._clear_buffer_kernel = context.compute_shader(read_shader_source('shader.ClearBuffer.comp'))


    def set_size(self, width, height):
        group_size_x = 8
        group_size_y = 8
        # group_size_z = 1

        self._width = width
        self._height = height
        self._num_cells = width * height
        self._num_groups_x = int(ceil(width / group_size_x))
        self._num_groups_y = int(ceil(height / group_size_y))

    def add_velocity(self, position, velocity, radius):
        tobytes = np.array(np.zeros(self._num_cells)).astype('f4').tobytes()
        print(tobytes)
        self._velocity_buffer = [self.context.buffer(tobytes),
                                 self.context.buffer(tobytes)]
        self._velocity_buffer[0].bind_to_storage_buffer(1)
        self._velocity_buffer[1].bind_to_storage_buffer(2)

        self._add_velocity_kernel['_Position'].value = position
        self._add_velocity_kernel['_Value'].value = velocity
        self._add_velocity_kernel['_Radius'].value = radius
        self._add_velocity_kernel['_Size'].value = (self._width, self._height)
        self._add_velocity_kernel.run(8,8,1)
        #self._flip_velocity_buffer()
        print(self._velocity_buffer[1].read())

    def _flip_velocity_buffer(self):
        tmp = self._velocity_buffer[self.VELOCITY_READ]
        self._velocity_buffer[self.VELOCITY_READ] = self._velocity_buffer[self.VELOCITY_WRITE]
        self._velocity_buffer[self.VELOCITY_WRITE] = tmp

    def delete(self):
        self._velocity_buffer[0].release()
        self._velocity_buffer[1].release()


if __name__ == '__main__':
    simulator = FluidSimulator(moderngl.create_standalone_context())
    simulator.set_size(10, 10)
    simulator.add_velocity((1, 1), (0.5, 0.5), 1.0)

    simulator.delete()
