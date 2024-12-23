OpenMM-Python-Force
===================

A plugin that enables OpenMM to receive energy values and optional energy gradients (with respect to coordinates) as `torch.Tensor`s or `numpy.ndarray` via the callback mechanism.

Quick Start
-----------

### Installation

```bash
# After cloning this project to src-dir,
# build in directory build-dir.
cmake -G "Unix Makefiles" \
      -S /path/to/src-dir/OpenMMPlugin \
      -B /path/to/build-dir \
      -DCOMPILE_TORCH_FORCE=1
make -C /path/to/build-dir install
make -C /path/to/build-dir PythonInstall
```

[Learn more about compilation and installation details.](doc/install.md)

### Usage

```python
import torch
from CallbackPyForce import Callable, TorchForce

class Model42(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, pos: torch.Tensor) -> torch.Tensor:
        return 0.5 * torch.sum(pos**2)

model42 = Model42()
model42 = model42.to("cuda")
cb = Callable(id(model42), Callable.RETURN_ENERGY)

# Import OpenMM packages.
# CUDA platform will be used.
# Add this force to an OpenMM System object.
cbtf = TorchForce(cb)
omm_system.addForce(cbtf)
```

[Find out more about the APIs.](doc/api.md)

Tests
-----

All tests passed in the following development environment
- Python 3.9.2
- torch 2.4.1+cu121
- gcc (Debian 10.2.1-6) 10.2.1 20210110

Unit tests depend on `make install` and `make wheel`. Fortunately, they do not require `make PythonInstall`. Tests can be executed using the following commands:
```bash
cd /path/to/build-dir
make install
make wheel
python3 -m pytest -vv -s test
```

If issues are identified in the unit tests and the installed binaries need to be removed, a shell script is provided: `src-dir/script/uninstall.sh`. This script does not delete any binary files. Instead, it lists the binaries for your reference.

Contribution
------------

Contributions are welcome! [Find out more about the authors and contributors.](CONTRIBUTING.md)

License
-------

[MIT License](LICENSE)
