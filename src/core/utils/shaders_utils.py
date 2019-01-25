from pathlib import Path

from jinja2 import Environment, FileSystemLoader

from src.core.common.constants import TemplateConstants


def read_shader_source(name):
    path = Path(__file__).parent.parent / "shaders"
    env = Environment(loader=FileSystemLoader(path), trim_blocks=True)

    env.get_template(name).render(
        NUM_THREADS=TemplateConstants.NUM_THREADS.value,
        VELOCITY_IN=TemplateConstants.VELOCITY_IN.value,
        VELOCITY_OUT=TemplateConstants.VELOCITY_OUT.value,
        PRESSURE_IN=TemplateConstants.PRESSURE_IN.value,
        PRESSURE_OUT=TemplateConstants.PRESSURE_OUT.value,
        VORTICITY=TemplateConstants.VORTICITY.value,
        DIVERGENCE=TemplateConstants.DIVERGENCE.value,
        OBSTACLES=TemplateConstants.OBSTACLES.value,
        GENERIC=TemplateConstants.GENERIC.value,
    )
