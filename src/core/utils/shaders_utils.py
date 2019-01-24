from pathlib import Path


# TODO: maybe templating to remove duplicated code?
def read_shader_source(name):
    path = Path(__file__).parent.parent / 'shaders' / name
    with open(path, 'r') as open_file:
        return open_file.read()
