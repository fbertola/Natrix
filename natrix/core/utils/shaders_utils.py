from ctypes import sizeof, c_float

from bgfx import bgfx, BGFX_BUFFER_COMPUTE_READ_WRITE


def create_buffer(length: int, dimensions: int, vertex_layout: bgfx.VertexLayout):
    return bgfx.create_dynamic_vertex_buffer(
        sizeof(c_float) * length * dimensions,
        vertex_layout,
        BGFX_BUFFER_COMPUTE_READ_WRITE,
    )
