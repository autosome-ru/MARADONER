# -*- coding: utf-8 -*-
__version__ = '0.33.0'
import importlib.metadata
import re


__min_reqs__ = [
            'typer>=0.13',
            'click>=8.0',
            'numpy>=2.1',
            'jax>=0.8',
            'jaxlib>=0.8',
            'matplotlib>=3.5',
            'pandas>=2.2',
            'scipy>=1.14',
            'statsmodels>=0.14',
            'datatable>=1.0.0' ,
            'dill>=0.3.9',
            'rich>=12.6.0',
            'tqdm>=4.0',
            'scikit-learn>=1.6',
            'threadpoolctl>=3.0',
            'joblib>=1.2',
            'tables>=3.10',
            'sympy>=1.12',
            'seaborn>=0.12',
            'pygam>=0.9'
           ]

def versiontuple(v):
    # Leading numeric components only: '2.1.0rc1' -> (2, 1, 0), '0.4.36.dev1+g12' -> (0, 4, 36).
    return tuple(map(int, re.findall(r'\d+', re.split(r'[^0-9.]', v, maxsplit=1)[0])))

def check_packages():
    """Raises ImportError listing every installed dependency that is older than the
    minimal version MARADONER is known to work with."""
    outdated = list()
    for req in __min_reqs__:
        name, _, ver = req.partition('>=')
        name, ver = name.strip(), ver.strip()
        if not ver:
            continue
        try:
            found = importlib.metadata.version(name)
            is_old = versiontuple(found) < versiontuple(ver)
        except (importlib.metadata.PackageNotFoundError, ValueError, TypeError):
            # Missing packages are reported by the import itself; unparsable versions
            # are given the benefit of the doubt.
            continue
        if is_old:
            outdated.append((name, ver, found))
    if outdated:
        items = ', '.join(f'{name} {found} (at least {ver} is needed)' for name, ver, found in outdated)
        cmd = 'pip install --upgrade ' + ' '.join(f'"{name}>={ver}"' for name, ver, _ in outdated)
        raise ImportError(f'MARADONER needs newer versions of the following packages: {items}. '
                          f'Upgrade them with: {cmd}')
