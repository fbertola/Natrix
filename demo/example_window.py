import ctypes
import sys
import os
import time

# noinspection PyPackageRequirements
import glfw

# noinspection PyUnresolvedReferences
from bgfx import bgfx, as_void_ptr

# noinspection PyPackageRequirements,PyProtectedMember
from glfw import _glfw as glfw_native


class ExampleWindow(object):
    def __init__(self, width, height, title):
        self.title = title
        self.height = height
        self.width = width
        self.ctx = None
        self.window = None
        self.fb_width = width
        self.fb_height = height
        self.hidpi = False

    def init(self, platform_data):
        pass

    def shutdown(self):
        pass

    def update(self, dt):
        pass

    def resize(self):
        pass

    def get_mouse_state(self):
        mouse_x, mouse_y = glfw.get_cursor_pos(self.window)

        if self.hidpi:
            mouse_x, mouse_y = mouse_x * 2, mouse_y * 2
        state_mbl = glfw.get_mouse_button(self.window, glfw.MOUSE_BUTTON_LEFT)
        state_mbm = glfw.get_mouse_button(self.window, glfw.MOUSE_BUTTON_MIDDLE)
        state_mbr = glfw.get_mouse_button(self.window, glfw.MOUSE_BUTTON_RIGHT)

        return (
            mouse_x,
            mouse_y,
            1
            if state_mbl == glfw.PRESS
            else 0 | 2
            if state_mbm == glfw.PRESS
            else 0 | 4
            if state_mbr == glfw.PRESS
            else 0,
        )

    # noinspection PyProtectedMember
    def run(self):
        glfw_native.glfwCreateWindow.argtypes = [
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_char_p,
        ]
        glfw_native.glfwCreateWindow.restype = ctypes.POINTER(glfw._GLFWwindow)
        glfw_native.glfwMakeContextCurrent.argtypes = [ctypes.POINTER(glfw._GLFWwindow)]
        glfw_native.glfwWindowShouldClose.argtypes = [ctypes.POINTER(glfw._GLFWwindow)]
        glfw_native.glfwWindowShouldClose.restype = ctypes.c_int

        glfw.init()

        glfw.window_hint(glfw.CLIENT_API, glfw.NO_API)
        glfw.window_hint(glfw.COCOA_RETINA_FRAMEBUFFER, glfw.TRUE)

        self.window = glfw.create_window(
            self.width, self.height, self.title, None, None
        )

        self.fb_width, self.fb_height = glfw.get_framebuffer_size(self.window)

        self.hidpi = self.fb_width != self.width or self.fb_height != self.height

        glfw.set_window_size_callback(self.window, self._handle_window_resize)

        handle, display = None, None

        if sys.platform == "darwin":
            glfw_native.glfwGetCocoaWindow.argtypes = [ctypes.POINTER(glfw._GLFWwindow)]
            glfw_native.glfwGetCocoaWindow.restype = ctypes.c_void_p
            handle = glfw_native.glfwGetCocoaWindow(self.window)
        elif sys.platform == "win32":
            glfw_native.glfwGetWin32Window.argtypes = [ctypes.POINTER(glfw._GLFWwindow)]
            glfw_native.glfwGetWin32Window.restype = ctypes.c_void_p
            handle = glfw_native.glfwGetWin32Window(self.window)
        elif sys.platform == "linux" and "WAYLAND_DISPLAY" not in os.environ:
            glfw_native.glfwGetX11Window.argtypes = [ctypes.POINTER(glfw._GLFWwindow)]
            glfw_native.glfwGetX11Window.restype = ctypes.c_void_p
            handle = glfw_native.glfwGetX11Window(self.window)
            display = glfw_native.glfwGetX11Display()
        elif sys.platform == "linux" and "WAYLAND_DISPLAY" in os.environ:
            glfw_native.glfwGetWaylandWindow.argtypes = [
                ctypes.POINTER(glfw._GLFWwindow)
            ]
            glfw_native.glfwGetWaylandWindow.restype = ctypes.c_void_p
            handle = glfw_native.glfwGetWaylandWindow(self.window)
            display = glfw_native.glfwGetWaylandDisplay()

        data = bgfx.PlatformData()
        data.ndt = display
        data.nwh = as_void_ptr(handle)
        data.context = None
        data.back_buffer = None
        data.back_buffer_ds = None

        self.init(data)

        last_time = None

        while not glfw.window_should_close(self.window):
            glfw.poll_events()

            now = time.perf_counter()
            if not last_time:
                last_time = now

            frame_time = now - last_time
            last_time = now

            self.update(frame_time)

        self.shutdown()
        glfw.terminate()

    def _handle_window_resize(self, window, width, height):
        self.width = width
        self.height = height

        self.fb_width, self.fb_height = glfw.get_framebuffer_size(window)

        self.resize()

    def _get_normalized_mouse_coords(self, mouse_x, mouse_y):
        return 1.0 - mouse_x / self.fb_width, 1.0 - mouse_y / self.fb_height
