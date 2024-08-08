# -*- coding: utf-8 -*-
__version__ = '0.1'
import importlib


__min_reqs__ = [
            'pip>=24.0',
            'typer>=0.6.1',
            'numpy>=1.21.0',
            'jax>=0.4.31',
            'jaxlib>=0.4.31',
            'matplotlib>=3.5.1',
            'pandas>=2.2.2',
            'scipy>=1.13',
            'statsmodels>=0.14',
            'datatable>=1.0.0' ,
            'dill>=0.3.8',
            'rich>=12.6.0',
           ]

def versiontuple(v):
    return tuple(map(int, (v.split("."))))

def check_packages():
    for req in __min_reqs__:
        try:
            module, ver = req.split(' @').split('>=')
            ver = versiontuple(ver)
            v = versiontuple(importlib.import_module(module).__version__)
        except (AttributeError, ValueError):
            continue
        if v < ver:
            raise ImportError(f'Version of the {module} package should be at least {ver} (found: {v}).')
