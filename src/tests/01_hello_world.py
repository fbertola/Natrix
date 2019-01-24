'''
    Renders a traingle that has all RGB combinations
'''

import moderngl

from src.core.fluid_simulator import FluidSimulator
from src.tests.example_window import Example, run_example


class TestFluidSimulator(Example):
    def __init__(self):
        self.ctx = moderngl.create_context()
        self.simulator = FluidSimulator(self.ctx)

    def render(self):
        self.ctx.viewport = self.wnd.viewport
        self.ctx.clear(0.2, 0.4, 0.7)


run_example(TestFluidSimulator)
