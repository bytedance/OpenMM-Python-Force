Compilation and Installation
============================

CMake Configuration Options
---------------------------

- `-DCMAKE_INSTALL_PREFIX=`: Specify installation directory. By default, shared libraries are installed to `/usr/local/openmm/lib`. Use this option if OpenMM is installed elsewhere.
- `-DCMAKE_BUILD_TYPE=`: Set build type. Defaults to `Release`. Other options: `MinSizeRel`, `RelWithDebInfo`, `Debug`.
- `-DPython3_EXECUTABLE=`: Specify an alternative `Python3` interpreter.
- `-DCOMPILE_TORCH_FORCE=0`: Do not compile the `TorchForce`.
- `-DUSE_CXX11_ABI=`: Defaults to `NOTSET`. Hopefully the default settings in your system will be compatible with the existing OpenMM and Torch libraries. If it is explicitly set to `0` or `1`, macro `_GLIBCXX_USE_CXX11_ABI` will be defined and set to `0` or `1`, respectively.

Python Extension Installation
-----------------------------

The plugin libraries and Python extensions are installed in separate directories. While `make PythonInstall` will install the Python extension to the default path without generating a wheel file, you can install to a custom path using:

```bash
cd /path/to/build-dir
make wheel
python3 -m pip install --prefix /new/path ./*.whl
```

For example, to install to a custom Python path like `/new/path/lib/python3.11/site-packages`, use the above commands.
