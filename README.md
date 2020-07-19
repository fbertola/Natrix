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

<p align="center">
  <img src="https://raw.githubusercontent.com/fbertola/Natrix/master/media/screenshot.png">
</p>

## Key Features

* Fast - leverages Compute Shaders to offload most of the calculations to the GPU.
* Built with BGFX rendering engine, supporting OpenGL, Vulkan, Metal and DirectX backends.
* Vorticity confinement. 
* Fluid obstacles.
* Poisson kernel.

## How To Use

To use this library, you'll need [Pipenv](https://github.com/pypa/pipenv) and the [BGFX Python wrapper](https://github.com/fbertola/bgfx-python).

From your command line:

```bash
# Clone this repository
$ git clone https://github.com/fbertola/Natrix .

# Go into the repository
$ cd Natrix

# Install dependencies
$ pipenv sync --dev
```

## Examples

In the [demo](https://github.com/fbertola/Natrix/tree/demo/demo) folder you will find a complete example, be sure to check it out. 

## Credits

This software uses the following open source packages:

- [BGFX](https://github.com/bkaradzic/bgfx)
- [BGFX Python Wrapper](https://github.com/fbertola/bgfx-python)

[License (BSD 2-clause)](https://raw.githubusercontent.com/fbertola/bgfx-python/master/LICENSE)
-----------------------------------------------------------------------

<a href="http://opensource.org/licenses/BSD-2-Clause" target="_blank">
<img align="right" src="http://opensource.org/trademarks/opensource/OSI-Approved-License-100x137.png">
</a>

    BSD 2-Clause License
    
    Copyright (c) 2020, Federico Bertola
    All rights reserved.
    
    Redistribution and use in source and binary forms, with or without
    modification, are permitted provided that the following conditions are met:
    
    1. Redistributions of source code must retain the above copyright notice, this
       list of conditions and the following disclaimer.
    
    2. Redistributions in binary form must reproduce the above copyright notice,
       this list of conditions and the following disclaimer in the documentation
       and/or other materials provided with the distribution.
    
    THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
    AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
    IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
    DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
    FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
    DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
    SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
    CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
    OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
    OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


---

> GitHub [@fbertola](https://github.com/fbertola)
