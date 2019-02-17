from pathlib import Path

import numpy as np
from jinja2 import Environment, FileSystemLoader
from moderngl import Context

from natrix.core.common.constants import TemplateConstants


def read_shader_source(name, root_path=None):
    path = Path(__file__).parent.parent / "shaders" if not root_path else root_path
    env = Environment(loader=FileSystemLoader(str(path.resolve())), trim_blocks=True)

    return env.get_template(name).render(
        NUM_THREADS=TemplateConstants.NUM_THREADS.value,
        VELOCITY_IN=TemplateConstants.VELOCITY_IN.value,
        VELOCITY_OUT=TemplateConstants.VELOCITY_OUT.value,
        PRESSURE_IN=TemplateConstants.PRESSURE_IN.value,
        PRESSURE_OUT=TemplateConstants.PRESSURE_OUT.value,
        VORTICITY=TemplateConstants.VORTICITY.value,
        DIVERGENCE=TemplateConstants.DIVERGENCE.value,
        OBSTACLES=TemplateConstants.OBSTACLES.value,
        GENERIC=TemplateConstants.GENERIC.value,
        PARTICLES_IN=TemplateConstants.PARTICLES_IN.value,
        PARTICLES_OUT=TemplateConstants.PARTICLES_OUT.value,
    )


def create_point_buffer(context: Context, length: int, default_value=0.0):
    data = [default_value for _ in range(length)]
    data_bytes = np.array(data).astype(np.float32).tobytes()

    return context.buffer(data_bytes)


def create_vector_buffer(
    context: Context, length: int, default_value_x=0.0, default_value_y=0.0
):
    data = [[default_value_x, default_value_y] for _ in range(length)]
    data_bytes = np.array(data).astype(np.float32).tobytes()

    return context.buffer(data_bytes)
