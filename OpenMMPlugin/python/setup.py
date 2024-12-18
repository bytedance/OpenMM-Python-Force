from setuptools import setup

module_name = "@MODULE_NAME@"
description = "@MODULE_NAME@"

swig_mod_name = f"_{module_name}"

setup(
    name=module_name,
    packages=["."],
    py_modules=[module_name],                   # ModuleName.py
    package_data={".": [f"{swig_mod_name}.*"]}  # _ModuleName.so
)
