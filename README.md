CUDA Examples
================================================================================

Some examples of CUDA applications ready for compiling and running.

- [1D Stencil](stencil) with naïve and optimized kernel.

Compiling
--------------------------------------------------------------------------------

Make sure you have a recent version o CMake (3.10+) installed. Then run:

```console
$ mkdir build && cd build
$ cmake -DCMAKE_BUILD_TYPE=Release
$ make -j
```

Author
--------------------------------------------------------------------------------

Created by [Gustavo Leite](https://github.com/leiteg) in October 2020.
