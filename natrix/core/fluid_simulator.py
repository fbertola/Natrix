from math import ceil

from moderngl import Context

from natrix.core.common.constants import TemplateConstants
from natrix.core.utils.shaders_utils import (
    read_shader_source,
    create_point_buffer,
    create_vector_buffer,
)


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

    _speed = 500.0
    _iterations = 50
    _dissipation = 1.0
    _vorticity = 0.0
    _viscosity = 0.1

    simulate = True

    def __init__(self, context: Context, width: int, height: int):
        self.context = context

        self._load_compute_kernels()
        self._set_size(width, height)
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
    def iterations(self):
        return self._iterations

    @iterations.setter
    def iterations(self, value):
        if value > 0:
            self._iterations = value
        else:
            raise ValueError("'Iterations' should be grater than zero")

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
    def vorticity(self):
        return self._vorticity

    @vorticity.setter
    def vorticity(self, value):
        if value >= 0:
            self._vorticity = value
        else:
            raise ValueError("'Vorticity' should be grater or equal than zero")

    @property
    def viscosity(self):
        return self._viscosity

    @viscosity.setter
    def viscosity(self, value):
        if value > 0:
            self._viscosity = value
        else:
            raise ValueError("'Viscosity' should be grater than zero")

    def add_velocity(self, position: tuple, velocity: tuple, radius: float):
        self._add_velocity_kernel["_Position"].value = position
        self._add_velocity_kernel["_Value"].value = velocity
        self._add_velocity_kernel["_Radius"].value = radius

        self._add_velocity_kernel.run(self._num_groups_x, self._num_groups_y, 1)
        self._flip_velocity_buffer()

    # position in normalised local space
    # radius in world space
    def add_circle_obstacle(self, position, radius, static=False):
        self._add_circle_obstacle_kernel["_Position"].value = position
        self._add_circle_obstacle_kernel["_Radius"].value = radius
        self._add_circle_obstacle_kernel["_Static"].value = 1 if static else 0

        self._add_circle_obstacle_kernel.run(self._num_groups_x, self._num_groups_y, 1)

    # points in normalised local space
    def add_triangle_obstacle(self, p1, p2, p3, static=False):
        self._add_triangle_obstacle_kernel["_P1"].value = p1
        self._add_triangle_obstacle_kernel["_P2"].value = p2
        self._add_triangle_obstacle_kernel["_P3"].value = p2
        self._add_triangle_obstacle_kernel["_Static"].value = 1 if static else 0

        self._add_triangle_obstacle_kernel.run(
            self._num_groups_x, self._num_groups_y, 1
        )

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
        self._advect_velocity_kernel["_Dissipation"].value = self.dissipation

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
            create_vector_buffer(self.context, self._num_cells),
            create_vector_buffer(self.context, self._num_cells),
        ]
        self._pressure_buffer = [
            create_point_buffer(self.context, self._num_cells),
            create_point_buffer(self.context, self._num_cells),
        ]
        self._divergence_buffer = create_point_buffer(self.context, self._num_cells)
        self._vorticity_buffer = create_point_buffer(self.context, self._num_cells)
        self._obstacles_buffer = create_vector_buffer(self.context, self._num_cells)

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
