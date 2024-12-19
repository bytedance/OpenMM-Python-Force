
APIs
====

Additional Python APIs can be found in the SWIG source files: `OpenMMPlugin/python/NumPyForce.i`, `OpenMMPlugin/python/TorchForce.i`.

API: Callable
-------------

```python
Callable(cpython_id: int, return_flag: int) -> "Callable"
    """
    cpython_id: The ID of the callable object.
    return_flag: Specifies the return type:
        - Callable.RETURN_ENERGY: Returns only energy.
        - Callable.RETURN_ENERGY_GRADIENT: Returns a tuple of (energy, gradient).
        - Callable.RETURN_ENERGY_FORCE: Returns a tuple of (energy, -gradient).
    """

add_kwarg(kwarg: str) -> None
    """
    Add a keyword argument in addition to the default coordinate argument.

    Currently supported kwargs:
    - cell (shape (3,3)): Representing [ax,ay,az,bx,by,bz,cx,cy,cz] when reshaped to (9,).
    - includeForces (bool): If the underlying callable supports energy-only calculations,
        using this kwarg to allow OpenMM to skip the computation of forces or gradients
        at its discretion.
    """
```

API: NumPyForce
---------------

The Callable object passed to `NumPyForce` accepts `fp64` inputs as `numpy.ndarray` and returns energy as a `float` (Python's `fp64`), or a tupe `(float, numpy.ndarray)` as (energy, gradient/gradient) where both are of type `fp64`.

```python
NumPyForce(callable, property_map: dict[str, str] = dict()) -> "NumPyForce"
    "The property_map is currently retained for compatibility with TorchForce."

setProperty(name: str, value: str) -> None
    "Retained for compatibility with TorchForce."

setForceGroup(group: int = 0) -> None
    "Standard OpenMM.Force API."
```

API: TorchForce
---------------

The Callable object passed to `TorchForce` accepts `torch.Tensor` objects as inputs and returns:
- Energy as a `torch.Tensor`.
- A tuple (energy, gradient/force) where both energy and gradient/force are torch tensors.

The `dtype` of the `torch.Tensor` objects aligns with OpenMM's `real` type. When the `CUDA` platform is used, the tensors remain on the GPU.

`TorchForce` supports the following properties for `property_map` and `setProperty` APIs:
- `{"useCUDAGraphs", "false"/"true"}`
- `{"CUDAGraphWarmupSteps", string(APositiveInteger)}`
