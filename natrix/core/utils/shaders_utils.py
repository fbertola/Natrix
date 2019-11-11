from pathlib import Path

import tempfile
import numpy as np
from bgfx import bgfx, shaderc, as_void_ptr, BGFX_BUFFER_COMPUTE_READ_WRITE
from jinja2 import Environment, FileSystemLoader

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
        OUT_IMAGE=TemplateConstants.OUT_IMAGE.value,
    )


def create_buffer(length: int, vertex_layout: bgfx.VertexLayout):
    data = [0.0 for _ in range(length)]
    data_bytes = np.array(data).astype(np.float32).tobytes()

    return bgfx.createDynamicVertexBuffer(len(data_bytes), vertex_layout, BGFX_BUFFER_COMPUTE_READ_WRITE)


def create_2d_buffer(length: int, vertex_layout: bgfx.VertexLayout):
    data = [[0.0, 0.0] for _ in range(length)]
    data_bytes = np.array(data).astype(np.float32).tobytes()

    return bgfx.createDynamicVertexBuffer(len(data_bytes), vertex_layout, BGFX_BUFFER_COMPUTE_READ_WRITE)
