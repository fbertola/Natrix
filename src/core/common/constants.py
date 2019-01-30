from enum import IntEnum

from decouple import config


class TemplateConstants(IntEnum):
    NUM_THREADS = config("NATRIX__NUM_THREADS", cast=int, default=8)
    VELOCITY_IN = 1
    VELOCITY_OUT = 2
    PRESSURE_IN = 3
    PRESSURE_OUT = 4
    VORTICITY = 5
    DIVERGENCE = 6
    OBSTACLES = 7
    GENERIC = 8
    PARTICLES_IN = 9
    PARTICLES_OUT = 10
