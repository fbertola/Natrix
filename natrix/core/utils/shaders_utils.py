from pathlib import Path

import numpy as np
from bgfx import bgfx, as_void_ptr, BGFX_BUFFER_COMPUTE_READ_WRITE, BGFX_BUFFER_COMPUTE_WRITE
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


def load_mem(path):
    with open(path, "rb") as f:
        read_data = f.read()
        size = len(read_data)
        memory = bgfx.copy(as_void_ptr(read_data), size)
        return memory


def load_shader(name, root_path=None):
    path = Path(__file__).parent.parent / "shaders" if not root_path else root_path

    shaders_path = {
        str(bgfx.RendererType.Noop): None,
        str(bgfx.RendererType.Direct3D9): "compiled/dx9",
        str(bgfx.RendererType.Direct3D11): "compiled/dx11",
        str(bgfx.RendererType.Direct3D12): "compiled/dx11",
        str(bgfx.RendererType.Gnm): "compiled/pssl",
        str(bgfx.RendererType.Metal): "compiled/Metal",
        str(bgfx.RendererType.Nvn): "compiled/nvn",
        str(bgfx.RendererType.OpenGL): "compiled/glsl",
        str(bgfx.RendererType.OpenGLES): "compiled/essl",
        str(bgfx.RendererType.Vulkan): "compiled/spirv",
    }
    print(str(bgfx.getRendererType()))
    complete_path = Path(path) / shaders_path.get(str(bgfx.getRendererType())) / name
    handle = bgfx.createShader(load_mem(complete_path))
    bgfx.setName(handle, name)

    return handle


def create_point_buffer(length: int, compute_vertex_decl: bgfx.VertexDecl):
    data = [0.0 for _ in range(length)]
    data_bytes = np.array(data).astype(np.float32).tobytes()

    return bgfx.createDynamicVertexBuffer(len(data_bytes), compute_vertex_decl, BGFX_BUFFER_COMPUTE_READ_WRITE)


def create_vector2_buffer(length: int, compute_vertex_decl: bgfx.VertexDecl):
    data = [[0.0, 0.0] for _ in range(length)]
    data_bytes = np.array(data).astype(np.float32).tobytes()

    return bgfx.createDynamicVertexBuffer(len(data_bytes), compute_vertex_decl, BGFX_BUFFER_COMPUTE_READ_WRITE)


def create_vector4_buffer(length: int, compute_vertex_decl: bgfx.VertexDecl):
    data = [[0.0, 0.0, 0.0, 0.0] for _ in range(length)]
    data_bytes = np.array(data).astype(np.float32).tobytes()

    return bgfx.createDynamicVertexBuffer(len(data_bytes), compute_vertex_decl, BGFX_BUFFER_COMPUTE_READ_WRITE)
