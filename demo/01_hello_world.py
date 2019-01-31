from pathlib import Path

import moderngl
import numpy as np
import pyglet.window.event
import random
from src.core.fluid_simulator import FluidSimulator
from src.core.particle_area import ParticleArea
from src.core.utils.shaders_utils import read_shader_source

root_path = Path(__file__).parent / "shaders"

ctx = moderngl.create_context()
wnd = pyglet.window.Window(width=512, height=512)

print(ctx.error)
print(wnd.context.get_info().get_version())

canvas = np.array([
    -1.0, -1.0,
    -1.0, 1.0,
    1.0, -1.0,
    -1.0, 1.0,
    1.0, 1.0,
    1.0, -1.0
]).astype('f4')

prog = ctx.program(
    vertex_shader=read_shader_source('demo.VertexShader.vert', root_path),
    fragment_shader=read_shader_source('demo.FragmentShader.frag', root_path)
)

pixels = np.zeros((512, 512), dtype='f4')
texture = ctx.texture((512, 512), 1, pixels.tobytes(), dtype='f4')
texture.filter = (moderngl.NEAREST, moderngl.NEAREST)
texture.swizzle = 'RRR1'
texture.use()

vbo = ctx.buffer(canvas.tobytes())
vao = ctx.simple_vertex_array(prog, vbo, 'pos')

fluid_simulator = FluidSimulator(ctx, 512, 512)
particle_area = ParticleArea(ctx, 512, 512)

fluid_simulator.vorticity = 10.0
fluid_simulator.viscosity = 0.0001

particle_area.dissipation = 0.999999


@wnd.event
def on_draw():
    wnd.clear()
    texture.write(particle_area.read_particles_buffer())
    vao.render()


def update(time_delta):
    vel_x = random.uniform(-0.08, 0.08)
    vel_y = random.uniform(0.0, 0.2)
    fluid_simulator.add_velocity((0.5, 0.01), (vel_x, vel_y), 60.0)
    particle_area.add_particles((0.5, 0.1), 40.0, 0.05)
    fluid_simulator.update(time_delta)
    particle_area.update(time_delta)
    print(pyglet.clock.get_fps())


pyglet.clock.schedule_interval(update, 1 / 60)

pyglet.app.run()
