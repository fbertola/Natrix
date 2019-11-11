import platform
from _ctypes import Structure
from ctypes import c_float, c_uint32


class PosVertex(Structure):
    _fields_ = [("m_x", c_float),
                ("m_y", c_float),
                ("m_z", c_float),
                ("m_abgr", c_uint32),
                ("m_u", c_float),
                ("m_v", c_float)]


def screen_space_quad(texture_width: int, texture_height: int, origin_bottom_left=False, width=1.0, height=1.0):
    if platform.system() == "Windows":
        s_texel_half = 0.5
    else:
        s_texel_half = 0.0

    zz = 0.0
    minx = -width
    maxx = width
    miny = 0.0
    maxy = height * 2.0
    texel_half_w = s_texel_half / texture_width
    texel_half_h = s_texel_half / texture_height
    minu = -1.0 + texel_half_w
    maxu = 1.0 + texel_half_w

    minv = texel_half_h
    maxv = 2.0 + texel_half_h

    if origin_bottom_left:
        temp = minv
        minv = maxv
        maxv = temp

        minv -= 1.0
        maxv -= 1.0

    vertices = (PosVertex * 3)(
        PosVertex(minx, miny, zz, 0xffffffff, minu, minv),
        PosVertex(maxx, miny, zz, 0xffffffff, maxu, minv),
        PosVertex(maxx, maxy, zz, 0xffffffff, maxu, maxv),
    )

    return vertices
