import ctypes
from array import array

from bgfx import bgfx, ImGui, ImVec2, ImVec4, as_void_ptr

from natrix.core.fluid_simulator import FluidSimulator
from natrix.core.particle_area import ParticleArea


class SampleData:
    m_values = []
    m_offset = 0
    m_min = 0.0
    m_max = 0.0
    m_avg = 0.0

    def __init__(self):
        self.reset()

    def reset(self):
        self.m_values = [0.0] * 100
        self.m_offset = 0
        self.m_min = 0.0
        self.m_max = 0.0
        self.m_avg = 0.0

    def push_sample(self, value: float):
        self.m_values[self.m_offset] = value

        min_val = float("inf")
        max_val = float("-inf")
        avg_val = 0.0

        # FIXME: da rivedere
        for val in self.m_values:
            min_val = min(min_val, val)
            max_val = max(max_val, val)
            avg_val += val

        self.m_min = min_val
        self.m_max = max_val
        self.m_avg = avg_val / 100.0

        self.m_offset = (self.m_offset + 1) % 100


def bar(width, max_width, height, color):
    style = ImGui.GetStyle()

    hovered_color = ImVec4(
        color.x + color.x * 0.1,
        color.y + color.y * 0.1,
        color.z + color.z * 0.1,
        color.w + color.w * 0.1,
    )

    ImGui.PushStyleColor(21, color)
    ImGui.PushStyleColor(22, hovered_color)
    ImGui.PushStyleColor(23, color)
    ImGui.PushStyleVar(12, 0.0)
    ImGui.PushStyleVar(14, ImVec2(0.0, style.ItemSpacing.y))

    item_hovered = False

    ImGui.Button("", ImVec2(width, height))
    item_hovered |= ImGui.IsItemHovered()

    ImGui.SameLine()
    ImGui.InvisibleButton("", ImVec2(max(1.0, max_width - width), height))
    item_hovered |= ImGui.IsItemHovered()

    ImGui.PopStyleVar(2)
    ImGui.PopStyleColor(3)

    return item_hovered


s_resourceColor = ImVec4(0.5, 0.5, 0.5, 1.0)
s_frame_time = SampleData()


def resource_bar(name, tooltip, num, _max, max_width, height):
    item_hovered = False

    ImGui.Text(f"{name}: {num:4d} / {_max:4d}")
    item_hovered |= ImGui.IsItemHovered()
    ImGui.SameLine()

    percentage = float(num) / float(_max)

    item_hovered |= bar(
        max(1.0, percentage * max_width), max_width, height, s_resourceColor
    )
    ImGui.SameLine()

    ImGui.Text(f"{(percentage * 100.0):5.2f}%")

    if item_hovered:
        ImGui.SetTooltip(f"{tooltip} {(percentage * 100.0):5.2f}%")


def show_properties_dialog(fluid_simulator: FluidSimulator, particle_system: ParticleArea):
    ImGui.SetNextWindowPos(ImVec2(20.0, 300.0), 1 << 2)
    ImGui.SetNextWindowSize(ImVec2(300.0, 400.0), 1 << 2)

    ImGui.Begin("\uf013 Properties")
    ImGui.TextWrapped("Simulation performances")

    ImGui.Separator()

    stats = bgfx.getStats()
    to_ms_cpu = 1000.0 / stats.cpuTimerFreq
    to_ms_gpu = 1000.0 / stats.gpuTimerFreq
    frame_ms = float(stats.cpuTimeEnd - stats.cpuTimeBegin)

    s_frame_time.push_sample(frame_ms * to_ms_cpu)

    frame_text_overlay = f"\uf063{s_frame_time.m_min:7.3f}ms, \uf062{s_frame_time.m_max:7.3f}ms\nAvg: {s_frame_time.m_avg:7.3f}ms, {(stats.cpuTimerFreq / frame_ms):6.2f} FPS"
    ImGui.PushStyleColor(40, ImVec4(0.0, 0.5, 0.15, 1.0))
    ImGui.PushItemWidth(-1)
    ImGui.PlotHistogram(
        "",
        array("f", s_frame_time.m_values)[0],
        100,
        s_frame_time.m_offset,
        frame_text_overlay,
        0.0,
        60.0,
        ImVec2(0.0, 45.0),
    )
    ImGui.PopItemWidth()
    ImGui.PopStyleColor()

    ImGui.Text(
        f"Submit CPU {(stats.cpuTimeEnd - stats.cpuTimeBegin) * to_ms_cpu:3.3f}, GPU {(stats.gpuTimeEnd - stats.gpuTimeBegin) * to_ms_gpu:3.3f} (L: {stats.maxGpuLatency})"
    )

    if stats.gpuMemoryMax > 0:
        ImGui.Text(f"GPU mem: {stats.gpuMemoryUsed} / {stats.gpuMemoryMax}")

    ImGui.Separator()

    vorticity = ImGui.Float(fluid_simulator.vorticity)
    viscosity = ImGui.Float(fluid_simulator.viscosity)
    speed = ImGui.Float(fluid_simulator.speed)
    iterations = ImGui.Int(fluid_simulator.iterations)
    borders = ImGui.Bool(fluid_simulator.has_borders)

    ImGui.Text("Fluid simulation parameters")

    if ImGui.SliderFloat("Vorticity", vorticity, 0.0, 10.0):
        fluid_simulator.vorticity = vorticity.value

    if ImGui.SliderFloat("Viscosity", viscosity, 0.000, 1.0):
        fluid_simulator.viscosity = viscosity.value

    if ImGui.SliderFloat("Speed", speed, 1.0, 1000.0):
        fluid_simulator.speed = speed.value
        particle_system.speed = speed.value

    if ImGui.SliderInt("Iteration", iterations, 10, 100):
        fluid_simulator.iterations = iterations.value

    if ImGui.Checkbox("Borders", borders):
        fluid_simulator.has_borders = borders.value

    ImGui.Separator()

    dissipation = ImGui.Float(particle_system.dissipation)

    ImGui.Text("Particles area parameters")

    if ImGui.SliderFloat("Dissipation", dissipation, 0.001, 1.0):
        particle_system.dissipation = dissipation.value

    ImGui.Separator()

    stop = ImGui.Bool(not fluid_simulator.simulate)

    if ImGui.Checkbox("Stop", stop):
        fluid_simulator.simulate = not stop.value
        particle_system.simulate = not stop.value

    ImGui.End()
