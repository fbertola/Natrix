<h1 align="center"> 
  <br>
  Natrix
  <br>
</h1>

<h4 align="center">Fast fluid simultation in Python.</h4>

<p align="center">
  <img src="https://img.shields.io/badge/python-v3.6+-blue.svg">
  <a href="https://travis-ci.com/fbertola/Natrix"><img src="https://travis-ci.com/fbertola/Natrix.svg?branch=master"></a>
  <img src="https://img.shields.io/badge/dependencies-up%20to%20date-brightgreen.svg">
  <a href="https://github.com/fbertola/Natrix/issues"><img src="https://img.shields.io/github/issues/fbertola/Natrix.svg"></a>
  <img src="https://img.shields.io/badge/contributions-welcome-orange.svg">
  <a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/license-MIT-blue.svg"></a>
</p>

<p align="center">
  <a href="#key-features">Key Features</a> •
  <a href="#how-to-use">How To Use</a> •
  <a href="#credits">Credits</a> •
  <a href="#license">License</a>
</p>

![screenshot](https://raw.githubusercontent.com/fbertola/Natrix/master/media/animation3.gif)

## Key Features

* Fast - leverages Compute Shaders to offloads most of the calculations to the GPU.
* Easy to integrate - does not require any particular framework.
* Vorticity confinement. 
* Fluid obstacles.
* Poisson kernel.
* Ready to use.

## How To Use

To use this library, you'll need [Pipenv](https://github.com/pypa/pipenv). From your command line:

```bash
# Clone this repository
$ git clone https://github.com/fbertola/Natrix .

# Go into the repository
$ cd Natrix

# Install dependencies
$ pipenv sync --dev
```

## Examples

You will find some examples in the `demo` folder, be sure to check them out before using Natrix.

## Credits

This software uses the following open source packages:

- [ModernGL](https://github.com/cprogrammer1994/ModernGL)
- [Jinja2](https://github.com/pallets/jinja)
- [Pyglet](https://bitbucket.org/pyglet/pyglet/wiki/Home)
- [Pygame](https://github.com/pygame/pygame)

## License

MIT

---

> GitHub [@fbertola](https://github.com/fbertola)
