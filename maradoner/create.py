from .utils import logger_print, warn, openers, InputError, stage_filename
from .dataset_filter import filter_lowexp
from .drist import DRIST
import multiprocessing
import scipy.stats as st
import datatable as dt
import pandas as pd
import numpy as np
import warnings
import dill
import json
import csv
import gzip
import os
import re


_TRANSFORMS = ('none', 'scale', 'ecdf', 'esf', 'drist', 'dristun')
# Intermediate files of a project that are produced after (and from) `create`.
_PROJECT_STAGES = ('init', 'fit', 'predict', 'fov', 'promoter_mean', 'promvar',
                   'old.fit', 'old.predict', 'old.fov')
_NUMERIC_STYPES = {dt.stype.bool8, dt.stype.int8, dt.stype.int16, dt.stype.int32, dt.stype.int64,
                   dt.stype.float32, dt.stype.float64}
_DECIMAL = re.compile(r'[+-]?(\d+\.\d*|\.\d+|\d+(\.\d*)?[eE][+-]?\d+)')


def _show(items, n: int = 5) -> str:
    """'a', 'b', 'c', ... and 7 more"""
    items = [str(i) for i in items]
    shown = ', '.join(f"'{i}'" for i in items[:n])
    if len(items) > n:
        shown += f', ... and {len(items) - n} more'
    return shown


def _resolve_n_jobs(n_jobs) -> int:
    """-1: all threads; a fraction in (0, 1): that share of the threads; a whole number >= 1: that many
    threads (never more than there are)."""
    cpu_count = multiprocessing.cpu_count()
    try:
        n = float(n_jobs)
    except (TypeError, ValueError):
        n = float('nan')
    if isinstance(n_jobs, bool) or not np.isfinite(n):
        n = float('nan')
    if n == -1:
        return cpu_count
    if 0 < n < 1:
        return max(1, int(n * cpu_count))
    if n >= 1 and n == int(n):
        return min(int(n), cpu_count)
    raise InputError('The number of threads (n_jobs) must be -1 (all threads), a fraction between 0 and 1 '
                     f'(of all threads) or a positive whole number of threads, but {n_jobs} was given.')


def drist_it(B: pd.DataFrame, Y: pd.DataFrame, test_chromosomes: list[str] = None,
             share_function: bool = False, optimizer='jacobi', verbose: bool = False):
    if test_chromosomes:
        pattern = re.compile(r'chr([0-9XYM]+|\d+)')

        test_chromosomes = set(test_chromosomes)
        mask = [pattern.search(p).group() in test_chromosomes for i, p in enumerate(Y.index)]
        mask = ~np.array(mask, dtype=bool)
    else:
        mask = np.ones(len(B), dtype=bool)
    Y = Y.values
    Y = Y - Y.mean(axis=1, keepdims=True)
    # `.values` of a pandas 3 frame is read-only: work on a copy and rebuild the frame.
    Bv = B.to_numpy(dtype=float, copy=True)
    Bt = Bv[mask, :]
    Y = Y[mask, :]
    drist = DRIST(max_iter=1000, verbose=verbose, share_function=share_function,
                  optimizer=optimizer)
    Bv[mask, :] = drist.fit_transform(Bt, Y)
    if not np.all(mask):
        Bv[~mask, :] = drist.transform(Bv[~mask, :])
    B = pd.DataFrame(Bv, index=B.index, columns=B.columns)

    B = B - B.min()
    return B


def _ecdf_column(v: np.ndarray) -> np.ndarray:
    """Empirical CDF of `v` at its own values (the arithmetic of scipy.stats.ecdf)."""
    return np.searchsorted(np.sort(v), v, side='right') / len(v)


def _column_stats(X: np.ndarray):
    """Per-column sample standard deviation (NaN if undefined) and maximal absolute value."""
    std = np.full(X.shape[1], np.nan)
    amax = np.zeros(X.shape[1])
    for j in range(X.shape[1]):
        c = X[:, j]
        if len(c) > 1:
            std[j] = c.std(ddof=1)
        amax[j] = np.abs(c).max() if len(c) else 0.0
    return std, amax


def _transform_array(X: np.ndarray, names: list, mode: str, Y: pd.DataFrame = None,
                     test_chromosomes: list[str] = None, verbose: bool = True):
    """Transform the columns of the float matrix `X` (promoters x motifs, modified in place or
    copied) and drop the motifs that end up constant. Returns the matrix and the names left."""
    if mode not in _TRANSFORMS:
        raise InputError(f'Unknown loading transformation "{mode}". Valid transformations: '
                         f'{", ".join(_TRANSFORMS)}.')
    names = list(names)
    stds, _ = _column_stats(X)
    drop_inds = (stds == 0) | np.isnan(stds)
    if drop_inds.any():
        dropped = [names[j] for j in np.flatnonzero(drop_inds)]
        warn(f'Dropping {len(dropped)} motif(s) whose loadings are constant (for example all zero) across the '
             f'{len(X)} promoters kept by the low-expression filter: {_show(dropped, 10)}.')
        X = X[:, ~drop_inds]
        names = [c for c, d in zip(names, drop_inds) if not d]
    # if not mode or mode == 'none':
    #     df[df < zero_cutoff] = 0
    #     df = (df - df.min(axis=None)) / (df.max(axis=None) - df.min(axis=None))
    if mode == 'scale':
        for j in range(X.shape[1]):
            c = X[:, j]
            lo, hi = c.min(), c.max()
            X[:, j] = (c - lo) / (hi - lo)
    elif mode == 'ecdf':
        for j in range(X.shape[1]):
            X[:, j] = _ecdf_column(X[:, j])
    elif mode in ('esf',):
        for j in range(X.shape[1]):
            v = 1 - _ecdf_column(X[:, j])
            # sf hits exactly 0 at the column maximum; clip it to the next smallest
            # value so that -log stays finite.
            positive = v[v > 0]
            t = positive.min() if len(positive) else 1.0 / len(v)
            v[v < t] = t
            X[:, j] = -np.log(v)
        # if mode == 'drist':
        #     df = drist_it(df, Y, test_chromosomes=test_chromosomes)
    elif mode.startswith('drist'):
        logger_print(f'Fitting the "{mode}" transformation of {X.shape[1]} motifs (this can take a while)...', verbose)
        df = pd.DataFrame(X, columns=names)
        df = drist_it(df, Y, test_chromosomes=test_chromosomes,
                      share_function=mode.endswith('un'))
        X = df.to_numpy(dtype=float)
    elif mode == 'none':
        pass
    elif mode:
        raise Exception('Unknown transformation mode ' + str(mode))
    # Re-test for degenerate columns *after* the transform: the ECDF/ESF map can turn a
    # column that did vary in the raw scores into a constant one. A motif whose scores
    # take only one or two distinct values among the retained promoters lands on a single
    # survival-function level (the top level has SF == 0 and is clipped up onto the next
    # one), so -log(SF) comes out the same for every promoter. Such a column is annihilated
    # by the promoter-wise centering H_p in `fit`, which leaves tau_k and mu_m,k
    # unidentified and B^T B singular.
    stds, amax = _column_stats(X)
    scale = np.where(amax == 0.0, 1.0, amax)
    drop_inds = (stds == 0) | np.isnan(stds) | (stds <= 1e-12 * scale)
    if drop_inds.any():
        dropped = [names[j] for j in np.flatnonzero(drop_inds)]
        hint = ' A column with only two distinct values (a 0/1 table, say) has no survival-function ' \
               'transform; use --loading-transform none or scale for such a matrix.' if mode == 'esf' else ''
        warn(f'Dropping {len(dropped)} motif(s) whose loadings are constant across '
             f'the retained promoters after the "{mode}" transform: {_show(dropped, 10)}.{hint}')
        X = X[:, ~drop_inds]
        names = [c for c, d in zip(names, drop_inds) if not d]
    return X, names


def transform_loadings(df, mode: str, zero_cutoff=1e-9, prom_inds=None, Y=None,
                       test_chromosomes: list[str] = None, verbose: bool = True):
    # Subset promoters *before* testing for constant columns: a motif can vary across
    # the full promoter set yet be constant among the promoters that survive the
    # low-expression filter, in which case its ECDF/ESF is degenerate.
    if prom_inds is not None:
        df = df.loc[prom_inds]
    X, names = _transform_array(df.to_numpy(dtype=float, copy=True), [str(c) for c in df.columns], mode,
                                Y=Y, test_chromosomes=test_chromosomes, verbose=verbose)
    return pd.DataFrame(X, index=df.index, columns=names)


# ------------------------------------------------------------------------------------------------
# Reading and validating input tables
# ------------------------------------------------------------------------------------------------

class _Table:
    """A table read from a file: first column = row names, header = column names."""
    def __init__(self, path, what, frame, rows, cols):
        self.path = path
        self.what = what
        self.frame = frame      # datatable Frame; column 0 holds the row names
        self.rows = rows        # list of str
        self.cols = cols        # list of str (names of columns 1..)


def _is_decimal(s: str) -> bool:
    return _DECIMAL.fullmatch(str(s).strip()) is not None


def _is_data_like(s: str) -> bool:
    """A decimal number with many digits (-4.104319916): a measurement, not an id or a name."""
    return _is_decimal(s) and sum(c.isdigit() for c in str(s)) >= 6


def _fread(path: str, what: str, n_jobs: int, **kwargs):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        try:
            frame = dt.fread(path, nthreads=n_jobs, **kwargs)
        except Exception as e:
            msg = re.sub(r'\s*<<.*?>>', '', str(e)).strip()
            msg = re.sub(r'\s*Set fill=True to ignore this error\.?', '', msg)
            msg = re.sub(r'\s*\(with sep=.*?\)', '', msg)
            raise InputError(f'The {what} "{path}" is not a regular table: {msg}. Every line must have the same '
                             'number of columns as the header, and there must be no blank lines.') from None
    return frame, [w for w in caught]


def _raw_header(path: str):
    """The first line of the file (bytes), or None if it cannot be inspected."""
    try:
        with open(path, 'rb') as f:
            magic = f.read(8)
        if magic.startswith((b'PK', b'BZh', b'\xfd7zXZ')):
            return None
        opener = gzip.open if magic[:2] == b'\x1f\x8b' else open
        with opener(path, 'rb') as f:
            head = f.read(4096)
            if b'\x00' in head:
                raise InputError('is binary')
            f.seek(0)
            return f.readline(1 << 24)
    except InputError:
        raise
    except (OSError, EOFError):
        return None


def _read_table(path, what: str, n_jobs: int, unique_rows: bool = True, col_label: str = 'column') -> _Table:
    """Read a delimited table with a header row and row names in the first column, check its
    structure and return it. `what` names the table in messages ("expression table")."""
    path = os.fspath(path)
    if not os.path.exists(path):
        raise InputError(f'The {what} "{path}" was not found.')
    if os.path.isdir(path):
        raise InputError(f'The {what} "{path}" is a directory, not a file.')
    if os.path.getsize(path) == 0:
        raise InputError(f'The {what} "{path}" is empty.')
    if path.lower().endswith(('.xls', '.xlsx', '.xlsm', '.ods')):
        raise InputError(f'The {what} "{path}" is a spreadsheet file. Export the sheet as a tab- or '
                         'comma-separated text file (for example "Save as CSV") and supply that file.')
    try:
        raw = _raw_header(path)
    except InputError:
        raise InputError(f'The {what} "{path}" is not a text table (it contains binary data). Supply a '
                         'tab- or comma-separated text file with a header row.') from None
    head, caught = _fread(path, what, n_jobs, header=True, max_nrows=1)
    if head.ncols == 0:
        raise InputError(f'The {what} "{path}" contains no table. Supply a tab- or comma-separated text file '
                         'with a header row.')
    # Row names are always read as text: ids such as 001 or 1.0 must not be turned into numbers.
    frame, caught = _fread(path, what, n_jobs, header=True, columns={head.names[0]: str})
    duplicated_by_reader = False
    for w in caught:
        text = str(w.message)
        if 'Duplicate column name' in text:
            duplicated_by_reader = True
        elif w.category.__name__ in ('DatatableWarning', 'IOWarning'):
            warn(f'While reading the {what} "{path}": {text}')
        else:
            warnings.warn_explicit(w.message, w.category, w.filename, w.lineno)
    if frame.ncols < 2:
        raise InputError(f'The {what} "{path}" has only one column. Expected a tab- or comma-separated table '
                         f'with row names in the first column followed by one column per {col_label}, '
                         'and a header row.')
    if frame.nrows < 1:
        raise InputError(f'The {what} "{path}" has a header row but no data rows.')
    cols = [str(c) for c in frame.names[1:]]
    fields = None
    if raw is not None:
        try:
            text = raw.decode('utf-8-sig').rstrip('\r\n')
        except UnicodeDecodeError:
            text = None
        if text is not None:
            for sep in ('\t', ',', ';', '|', ' '):
                try:
                    cand = next(csv.reader([text], delimiter=sep))
                except (csv.Error, StopIteration):
                    continue
                if len(cand) == frame.ncols:
                    fields = cand[1:]
                    break
                if len(cand) == frame.ncols - 1:
                    fields = cand
                    break
    if fields is not None:
        empty = [i + 1 for i, c in enumerate(fields) if not c.strip()]
        if empty:
            raise InputError(f'The header of the {what} "{path}" has empty {col_label} names (column '
                             f'{", ".join(map(str, empty[:5]))}). Every column needs a name.')
        seen, dups = set(), list()
        for c in fields:
            if c in seen and c not in dups:
                dups.append(c)
            seen.add(c)
        if dups:
            raise InputError(f'The header of the {what} "{path}" contains duplicated {col_label} names: '
                             f'{_show(dups)}. Names must be unique.')
    elif duplicated_by_reader:
        raise InputError(f'The header of the {what} "{path}" contains duplicated {col_label} names. '
                         'Names must be unique.')
    decimals = [c for c in cols if _is_data_like(c)]
    if len(decimals) > len(cols) / 2:
        raise InputError(f'The {what} "{path}" seems to have no header row: its first line holds numbers '
                         f'({_show(decimals, 3)}) where {col_label} names are expected. The first line must '
                         f'contain the {col_label} names.')
    rows = frame[:, 0].to_list()[0]
    if all(r is not None and _is_data_like(r) for r in rows):
        raise InputError(f'The first column of the {what} "{path}" holds numbers ({_show(rows, 3)}) rather than '
                         f'row names. The first column must contain the row names, followed by one column per '
                         f'{col_label}.')
    missing = [i + 1 for i, r in enumerate(rows) if r is None or not str(r).strip()]
    if missing:
        raise InputError(f'The {what} "{path}" has rows without a name in the first column (data row '
                         f'{", ".join(map(str, missing[:5]))}). The first column must contain the row names.')
    rows = [str(r) for r in rows]
    if unique_rows:
        seen, dups = set(), list()
        for r in rows:
            if r in seen and r not in dups:
                dups.append(r)
            seen.add(r)
        if dups:
            raise InputError(f'The {what} "{path}" has {len(dups)} row name(s) that occur more than once in the '
                             f'first column: {_show(dups)}. Row names must be unique.')
    return _Table(path, what, frame, rows, cols)


def _matrix(table: _Table, rows=None, cols=None, row_word: str = 'row', col_word: str = 'column') -> np.ndarray:
    """The numbers of `table` (selected rows / columns, as indices into table.rows / table.cols) as a
    float64 array. Cells that are not finite numbers raise an InputError naming them."""
    frame = table.frame
    cidx = list(range(1, frame.ncols)) if cols is None else [c + 1 for c in cols]
    sub = frame[slice(None) if rows is None else list(rows), cidx]
    row_names = (lambda i: table.rows[i]) if rows is None else (lambda i: table.rows[rows[i]])
    col_names = (lambda j: table.cols[j]) if cols is None else (lambda j: table.cols[cols[j]])
    if all(s in _NUMERIC_STYPES for s in sub.stypes):
        X = np.asarray(sub.to_numpy(), dtype=float)
        if not X.flags.writeable:
            X = X.copy()
    else:
        df = sub.to_pandas()
        X = np.empty(df.shape, dtype=float, order='F')
        bad = list()
        for j in range(df.shape[1]):
            s = df.iloc[:, j]
            if s.dtype.kind in 'biuf':
                X[:, j] = s.to_numpy(dtype=float)
                continue
            num = pd.to_numeric(s, errors='coerce')
            blank = s.astype(str).str.strip() == ''
            wrong = np.flatnonzero((num.isna() & s.notna() & ~blank).to_numpy())
            bad.extend((i, j, s.iloc[i]) for i in wrong)
            X[:, j] = num.to_numpy(dtype=float)
        if bad:
            i, j, v = bad[0]
            hint = ' Decimal commas are not supported, use a dot.' if re.fullmatch(r'\s*-?\d+,\d+\s*', str(v)) else ''
            examples = '; '.join(f'"{v}" at {row_word} "{row_names(i)}", {col_word} "{col_names(j)}"'
                                 for i, j, v in bad[:3])
            raise InputError(f'The {table.what} "{table.path}" contains {len(bad)} value(s) that are not '
                             f'numbers: {examples}.{hint} All values must be numeric.')
    if not np.isfinite(X.sum()):
        nan = np.isnan(X)
        inf = np.isinf(X)
        n_nan, n_inf = int(nan.sum()), int(inf.sum())
        if n_nan or n_inf:
            bad = nan | inf
            empty_cols = np.flatnonzero(bad.all(axis=0)) if X.shape[0] > 1 else []
            empty_rows = np.flatnonzero(bad.all(axis=1)) if X.shape[1] > 1 else []
            where = list()
            if len(empty_cols):
                where.append(f'{col_word}(s) without any valid value: {_show([col_names(j) for j in empty_cols], 3)}')
            if len(empty_rows):
                where.append(f'{row_word}(s) without any valid value: {_show([row_names(i) for i in empty_rows], 3)}')
            if not where:
                where.append('e.g. ' + '; '.join(f'{row_word} "{row_names(i)}", {col_word} "{col_names(j)}" '
                                                 f'({"missing" if nan[i, j] else "infinite"})'
                                                 for i, j in np.argwhere(bad)[:3]))
            counts = ' and '.join(x for x in (f'{n_nan} missing' if n_nan else '', f'{n_inf} infinite' if n_inf else '') if x)
            hint = ' Infinite values usually come from the logarithm of 0: use log(x + 1) or similar.' if n_inf else ''
            raise InputError(f'The {table.what} "{table.path}" contains {counts} value(s), {"; ".join(where)}. '
                             'MARADONER does not impute values: fill in or remove the affected '
                             f'{row_word}s or {col_word}s.{hint}')
    return X


def _no_overlap_hint(a: _Table, b: _Table) -> str:
    """Possible reasons why the row names of two tables have nothing in common."""
    hints = list()
    ra, rb = set(a.rows), set(b.rows)
    if {r.strip() for r in ra} & {r.strip() for r in rb}:
        hints.append('the ids differ in leading or trailing blanks')
    elif {r.lower() for r in ra} & {r.lower() for r in rb}:
        hints.append('the ids differ in upper/lower case')
    elif set(a.cols) & rb or set(b.cols) & ra:
        hints.append('one of the tables seems to be transposed (rows and columns exchanged)')
    return (' Possible reasons: ' + '; '.join(hints) + '.') if hints else ' Are the first columns of both files ' \
           'the promoter ids, spelled identically?'


# ------------------------------------------------------------------------------------------------
# Sample groups
# ------------------------------------------------------------------------------------------------

def _no_duplicate_keys(pairs, path: str = ''):
    keys = [k for k, _ in pairs]
    dups = sorted({k for k in keys if keys.count(k) > 1})
    if dups:
        raise InputError(f'Group name(s) {_show(dups)} occur more than once in the sample groups file "{path}". '
                         'Group names must be unique.')
    return dict(pairs)


def _load_groups_file(path: str) -> dict:
    if not os.path.exists(path):
        raise InputError(f'The sample groups file "{path}" was not found.')
    if os.path.isdir(path):
        raise InputError(f'The sample groups file "{path}" is a directory, not a file.')
    try:
        with open(path, 'rb') as f:
            text = f.read().decode('utf-8-sig')
    except UnicodeDecodeError:
        raise InputError(f'The sample groups file "{path}" is not a text file.') from None
    stripped = text.strip()
    if not stripped:
        raise InputError(f'The sample groups file "{path}" is empty. It must define at least one group.')
    if stripped[0] in '{[':
        try:
            return json.loads(stripped, object_pairs_hook=lambda pairs: _no_duplicate_keys(pairs, path))
        except json.JSONDecodeError as e:
            raise InputError(f'The sample groups file "{path}" looks like JSON, but is not valid JSON: {e.msg} '
                             f'(line {e.lineno}, column {e.colno}).') from None
    groups = dict()
    for lineno, line in enumerate(text.splitlines(), 1):
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        items = line.split()
        if items[0] in groups:
            raise InputError(f'Group "{items[0]}" is defined twice in the sample groups file "{path}" '
                             f'(line {lineno}). Group names must be unique.')
        hint = ' If the names are separated by commas, use spaces or tabs instead.' if len(items) == 1 and ',' in items[0] else ''
        if len(items) == 1:
            raise InputError(f'Group "{items[0]}" (line {lineno} of the sample groups file "{path}") has no '
                             'samples. Each line must be a group name followed by the names of its samples, '
                             f'separated by spaces or tabs.{hint}')
        groups[items[0]] = items[1:]
    return groups


def read_groups(sample_groups, sample_names: list, subset: bool = False, row_names: list = None) -> dict:
    """Validate sample groups against the sample names of the expression table. `sample_groups` is a
    path (JSON, or text with one `group sample sample ...` line per group) or a dict group -> samples.
    Returns group -> list of its samples (in the order of the table); the input is not modified."""
    if isinstance(sample_groups, (str, os.PathLike)):
        path = os.fspath(sample_groups)
        groups = _load_groups_file(path)
        source = f'the sample groups file "{path}"'
    else:
        groups = sample_groups
        source = 'the sample groups'
    if not isinstance(groups, dict):
        raise InputError(f'{source[0].upper() + source[1:]} must define a mapping from group names to lists of '
                         f'sample names (for example {{"group_1": ["sample_1", "sample_2"]}}), but holds a '
                         f'{type(groups).__name__}.')
    if not groups:
        raise InputError(f'{source[0].upper() + source[1:]} defines no groups.')
    clean = dict()
    for name, members in groups.items():
        if not isinstance(name, str) or not name.strip():
            raise InputError(f'{source[0].upper() + source[1:]} has a group without a name.')
        if isinstance(members, str) or not isinstance(members, (list, tuple)):
            raise InputError(f'Group "{name}" in {source} must be a list of sample names, but is a '
                             f'{type(members).__name__}.')
        if not all(isinstance(m, str) for m in members):
            other = next(m for m in members if not isinstance(m, str))
            raise InputError(f'Group "{name}" in {source} contains {other!r}, which is not a sample name. '
                             'Sample names must be strings (put numeric names in quotes).')
        if not members:
            raise InputError(f'Group "{name}" in {source} has no samples.')
        twice = sorted({m for m in members if list(members).count(m) > 1})
        if twice:
            raise InputError(f'Sample(s) {_show(twice)} are listed more than once in group "{name}" of {source}.')
        clean[name] = list(members)
    owner = dict()
    for name, members in clean.items():
        for m in members:
            if m in owner:
                raise InputError(f'Sample "{m}" belongs to the groups "{owner[m]}" and "{name}" in {source}. '
                                 'Each sample can be in one group only.')
            owner[m] = name
    known = set(sample_names)
    absent = [m for m in owner if m not in known]
    if absent and not subset:
        transposed = ' The samples are listed among the rows of the table: it must have one column per sample.' \
            if row_names is not None and len(set(absent) & set(row_names)) > len(absent) / 2 else ''
        raise InputError(f'{len(absent)} sample(s) of {source} are not columns of the expression table: '
                         f'{_show(absent)}.{transposed} Check the spelling, or use --sample-groups-subset to '
                         'ignore samples that are absent from the expression table.')
    if absent:
        warn(f'{len(absent)} sample(s) of {source} are not in the expression table and were ignored: '
             f'{_show(absent)}.')
        clean = {n: [m for m in members if m in known] for n, members in clean.items()}
        empty = [n for n, members in clean.items() if not members]
        if empty:
            warn(f'Group(s) {_show(empty)} have no samples in the expression table and were removed.')
            clean = {n: members for n, members in clean.items() if members}
        if not clean:
            raise InputError(f'None of the samples of {source} are columns of the expression table (for '
                             f'example {_show(absent, 3)} versus {_show(sample_names, 3)}).')
    used = {m for members in clean.values() for m in members}
    unused = [s for s in sample_names if s not in used]
    if unused:
        warn(f'{len(unused)} of {len(sample_names)} samples of the expression table are in no group and '
             f'were dropped: {_show(unused, 10)}.')
    position = {s: i for i, s in enumerate(sample_names)}
    return {n: sorted(members, key=position.__getitem__) for n, members in clean.items()}


# ------------------------------------------------------------------------------------------------
# Covariates
# ------------------------------------------------------------------------------------------------

def build_covariates(covariates_filename: str, sample_names: list, n_jobs: int = 1,
                     verbose: bool = True):
    """Read a covariate/metadata table and build the design matrix X of shape
    (c, s) aligned to ``sample_names``.

    The first column of the file is treated as the sample identifier (index).
    Remaining columns are covariates. Numeric covariates are z-scored; categorical
    covariates are one-hot encoded (first level dropped) and centred. An intercept
    row of ones is always prepended, so the returned X always has at least one row.
    Every sample must be present in the table. Missing cells are imputed with the column
    mean (numeric) / most-frequent level (categorical), constant columns are dropped, and
    covariates that are linear combinations of the others are rejected.

    Returns (X, covariate_names).
    """
    path = os.fspath(covariates_filename)
    sample_names = [str(s) for s in sample_names]
    table = _read_table(path, 'covariate table', _resolve_n_jobs(n_jobs), unique_rows=False, col_label='covariate')
    df = table.frame[:, 1:].to_pandas()
    df.columns = table.cols
    ids = pd.Index(table.rows)
    wanted = set(sample_names)
    found = [i for i in dict.fromkeys(table.rows) if i in wanted]
    if not found:
        raise InputError(f'None of the {len(sample_names)} sample ids of the project (for example '
                         f'{_show(sample_names, 3)}) occur in the first column '
                         f'of the covariate table "{path}" (for example {_show(table.rows, 3)}). The first '
                         'column must contain the sample ids.')
    missing = [s for s in sample_names if s not in set(found)]
    if missing:
        raise InputError(f'{len(missing)} of {len(sample_names)} samples are absent from the first column of '
                         f'the covariate table "{path}": {_show(missing)}. Add a row for each of them.')
    extra = [i for i in dict.fromkeys(table.rows) if i not in wanted]
    if extra:
        warn(f'{len(extra)} row(s) of the covariate table "{path}" belong to no sample of the project and '
             f'were ignored: {_show(extra)}.')
    dup = ids.duplicated(keep=False) & ids.isin(sample_names)
    if dup.any():
        sub = df[dup].copy()
        sub.index = ids[dup]
        conflicting = [k for k, g in sub.groupby(level=0, sort=False) if len(g.drop_duplicates()) > 1]
        if conflicting:
            raise InputError(f'The covariate table "{path}" lists {len(conflicting)} sample(s) more than once '
                             f'with different values: {_show(conflicting)}. Keep one row per sample.')
        logger_print(f'The covariate table lists {int(ids[dup].nunique())} sample(s) several times with identical '
                     'values; the repeats were ignored.', verbose)
    # Metadata files may contain (many) duplicate rows; keep the first occurrence
    # per sample id.
    first = ~ids.duplicated(keep='first')
    df = df[first]
    df.index = ids[first]
    # Align to the samples of the project.
    df = df.reindex(sample_names)
    rows = [np.ones(len(sample_names), dtype=float)]
    names = ['intercept']
    imputed, constant, empty = list(), list(), list()
    for col in df.columns:
        series = df[col]
        if series.dtype.kind not in 'biuf':
            # an empty cell of a text column is read as an empty string
            series = series.where(series.astype(str).str.strip() != '', np.nan)
        if series.dtype.kind in 'biuf':
            numeric = series.astype(float)
        else:
            numeric = pd.to_numeric(series, errors='coerce')
            n_present, n_numeric = int(series.notna().sum()), int(numeric.notna().sum())
            if n_present and n_numeric and n_numeric < n_present:
                as_text = series[series.notna() & numeric.isna()].iloc[0]
                as_number = series[numeric.notna()].iloc[0]
                raise InputError(f'Covariate "{col}" in "{path}" mixes numbers (for example {as_number}) and '
                                 f'text (for example "{as_text}"). Use numbers only for a numeric covariate '
                                 'and text only for a categorical one.')
            if n_numeric < n_present:
                numeric = None
        if series.notna().sum() == 0:
            empty.append(str(col))
            continue
        if numeric is not None:
            v = numeric.astype(float)
            if np.isinf(v).any():
                raise InputError(f'Covariate "{col}" in "{path}" contains infinite values.')
            if v.isna().any():
                imputed.append((str(col), int(v.isna().sum())))
            v = v.fillna(v.mean())
            std = v.std()
            if std == 0 or not np.isfinite(std):
                constant.append(str(col))
                continue
            v = (v - v.mean()) / std
            rows.append(v.values.astype(float))
            names.append(str(col))
        else:
            s = series.astype('object')
            mode = s.dropna()
            if s.isna().any():
                imputed.append((str(col), int(s.isna().sum())))
            mode = mode.mode().iloc[0] if len(mode) else ''
            s = s.fillna(mode)
            dummies = pd.get_dummies(s, prefix=str(col), drop_first=True)
            if dummies.shape[1] == 0:
                constant.append(str(col))
            for dcol in dummies.columns:
                v = dummies[dcol].astype(float)
                if v.std() == 0:
                    continue
                v = v - v.mean()
                rows.append(v.values.astype(float))
                names.append(str(dcol))
    if empty:
        warn(f'Covariate(s) {_show(empty)} of "{path}" have no values at all and were dropped.')
    if constant:
        warn(f'Covariate(s) {_show(constant)} of "{path}" are constant over the samples of the project and '
             'were dropped.')
    if imputed:
        warn('Missing values of the covariate table were imputed (column mean for numeric covariates, most '
             'frequent level for categorical ones): ' + ', '.join(f'"{c}" ({k} of {len(sample_names)} samples)'
                                                                   for c, k in imputed) + '.')
    if len(names) == 1:
        warn(f'No usable covariate is left in "{path}"; the project is created with the intercept only.')
    repeated = sorted({n for n in names if names.count(n) > 1})
    if repeated:
        raise InputError(f'Covariate names are not unique after encoding categorical covariates: '
                         f'{_show(repeated)}. Rename the columns of "{path}".')
    X = np.vstack(rows)
    if X.shape[0] > X.shape[1]:
        raise InputError(f'There are {X.shape[0] - 1} covariates (plus the intercept) for only {X.shape[1]} '
                         f'samples, so their effects cannot be estimated. Remove covariates from "{path}".')
    accepted, dependent = [0], list()
    for i in range(1, X.shape[0]):
        A = X[accepted].T
        coef = np.linalg.lstsq(A, X[i], rcond=None)[0]
        if np.linalg.norm(X[i] - A @ coef) <= 1e-8 * max(np.linalg.norm(X[i]), 1.0):
            parts = [names[accepted[k]] for k in np.flatnonzero(np.abs(coef) > 1e-6)]
            dependent.append(f'"{names[i]}" is a combination of {", ".join(chr(34) + p + chr(34) for p in parts)}')
        else:
            accepted.append(i)
    if dependent:
        raise InputError(f'The covariates of "{path}" are linearly dependent, so their effects cannot be told '
                         f'apart: {"; ".join(dependent[:5])}. Remove one covariate of each such set.')
    return X, names


# ------------------------------------------------------------------------------------------------
# Project
# ------------------------------------------------------------------------------------------------

def _read_motif_names(path) -> list:
    path = os.fspath(path)
    if not os.path.exists(path):
        raise InputError(f'The motif names file "{path}" was not found.')
    if os.path.isdir(path):
        raise InputError(f'The motif names file "{path}" is a directory, not a file.')
    try:
        with open(path, 'rb') as f:
            text = f.read().decode('utf-8-sig')
    except UnicodeDecodeError:
        raise InputError(f'The motif names file "{path}" is not a text file.') from None
    names = list(dict.fromkeys(text.split()))
    if not names:
        raise InputError(f'The motif names file "{path}" is empty. It must list the motifs to keep.')
    return names


def _require_file(path: str, label: str):
    if os.path.isdir(path):
        raise InputError(f'{label} {path} is a directory, not a file.')
    if not os.path.isfile(path):
        raise InputError(f'{label} {path} not found.')


def _check_output_folder(project_name: str, plot_filename, dump: bool = True):
    paths = list()
    if dump:
        base = os.path.basename(project_name)
        if not base or base in ('.', '..'):
            raise InputError(f'"{project_name}" is not a valid project name: it must end with a file name '
                             '(project files are written as "<name>.init.raw" and so on).')
        if len(base) > 200:
            raise InputError(f'The project name is too long ({len(base)} characters): project files are named '
                             'after it, and file names are limited to 255 characters.')
        paths.append(project_name)
    if plot_filename:
        paths.append(os.fspath(plot_filename))
    for path in paths:
        folder = os.path.dirname(path)
        if folder and not os.path.isdir(folder):
            raise InputError(f'The directory "{folder}" of the project "{project_name}" does not exist. '
                             'Create it first or choose another project name.')
        if not os.access(folder or '.', os.W_OK | os.X_OK):
            raise InputError(f'The directory "{folder or "."}" of the project "{project_name}" is not '
                             'writable.')


def create_project(project_name: str, promoter_expression_filename: str, loading_matrix_filenames: list[str],
                   motif_expression_filenames=None, loading_matrix_transformations=None, sample_groups=None, motif_postfixes=None,
                   promoter_filter_lowexp_cutoff=0.9, promoter_filter_plot_filename=None, promoter_filter_max=True,
                   sample_groups_subset=False,  motif_names_filename=None, covariates_filename=None,
                   n_jobs:float = 0.5, compression='raw', dump=True, verbose=True):
    project_name = os.fspath(project_name)
    # ---- arguments (nothing is read yet) ----
    if compression not in openers:
        raise InputError(f'Unknown compression "{compression}". Valid values: {", ".join(openers)}.')
    n_jobs = _resolve_n_jobs(n_jobs)
    w = promoter_filter_lowexp_cutoff
    if not (isinstance(w, (int, float, np.integer, np.floating)) and 0 < w <= 1):
        raise InputError('The low-expression filter parameter w (--filter-lowexp-w) must be a number with '
                         '0 < w <= 1 (a promoter is dropped iff its posterior probability of belonging to the '
                         f'low-expression component exceeds w; w = 1 keeps all promoters), but {w} was given.')
    promoter_expression_filename = os.fspath(promoter_expression_filename)
    _require_file(promoter_expression_filename, 'Promoter expression file')
    if isinstance(loading_matrix_filenames, (str, os.PathLike)):
        loading_matrix_filenames = [loading_matrix_filenames]
    loading_matrix_filenames = [os.fspath(f) for f in loading_matrix_filenames]
    if not loading_matrix_filenames:
        raise InputError('No loading matrix was given.')
    n_mat = len(loading_matrix_filenames)
    for mx_name in loading_matrix_filenames:
        _require_file(mx_name, 'Loading matrix file')
    if motif_expression_filenames:
        if isinstance(motif_expression_filenames, (str, os.PathLike)):
            motif_expression_filenames = [motif_expression_filenames]
        motif_expression_filenames = [os.fspath(f) for f in motif_expression_filenames]
        for exp_name in motif_expression_filenames:
            _require_file(exp_name, 'Motif expression file')
        if len(motif_expression_filenames) != n_mat:
            raise InputError(f'{len(motif_expression_filenames)} motif expression table(s) were given for {n_mat} '
                             'loading matrix/matrices: give exactly one per loading matrix (in the same order).')
    if loading_matrix_transformations is None:
        loading_matrix_transformations = ['esf']
    elif isinstance(loading_matrix_transformations, str):
        loading_matrix_transformations = [loading_matrix_transformations]
    loading_matrix_transformations = [getattr(t, 'value', t) for t in loading_matrix_transformations]
    if len(loading_matrix_transformations) == 1:
        loading_matrix_transformations = [loading_matrix_transformations[0]] * n_mat
    elif len(loading_matrix_transformations) != n_mat:
        raise InputError(f'Total number of loading matrices is {n_mat}, but the number of transformations is '
                         f'{len(loading_matrix_transformations)}: give one transformation for all matrices or '
                         'exactly one per matrix.')
    for mode in loading_matrix_transformations:
        if mode not in _TRANSFORMS:
            raise InputError(f'Unknown loading transformation "{mode}". Valid transformations: '
                             f'{", ".join(_TRANSFORMS)}.')
    if isinstance(motif_postfixes, str):
        motif_postfixes = [motif_postfixes]
    postfixes = [None] * n_mat
    if motif_postfixes:
        if len(motif_postfixes) != n_mat:
            raise InputError(f'{len(motif_postfixes)} loading postfix(es) were given for {n_mat} loading '
                             'matrix/matrices: give exactly one postfix per loading matrix (in the same '
                             'order), or none.')
        postfixes = [str(p) for p in motif_postfixes]
    _check_output_folder(project_name, promoter_filter_plot_filename, dump)
    wanted_motifs = _read_motif_names(motif_names_filename) if motif_names_filename is not None else None

    # ---- expression and samples ----
    logger_print('Reading dataset...', verbose)
    expr = _read_table(promoter_expression_filename, 'promoter expression table', n_jobs, col_label='sample')
    if sample_groups:
        groups_by_name = read_groups(sample_groups, expr.cols, subset=sample_groups_subset, row_names=expr.rows)
        used = {m for members in groups_by_name.values() for m in members}
    else:
        groups_by_name = None
        used = set(expr.cols)
    sample_idx = [i for i, c in enumerate(expr.cols) if c in used]
    sample_names = [expr.cols[i] for i in sample_idx]
    Y_all = _matrix(expr, cols=sample_idx, row_word='promoter', col_word='sample')
    if (Y_all >= 0).all() and (np.array_equal(Y_all, np.round(Y_all)) or Y_all.max() > 50):
        warn(f'The expression values in "{expr.path}" look like raw counts or intensities (largest value '
             f'{Y_all.max():g}), but MARADONER expects expression on a log scale, e.g. log2(TPM + 1).')
    if covariates_filename:
        logger_print('Building covariate design matrix...', verbose)
        covariates, covariate_names = build_covariates(covariates_filename, sample_names,
                                                       n_jobs=n_jobs, verbose=verbose)
    else:
        covariates, covariate_names = None, None

    # ---- loading matrices ----
    tables = [_read_table(f, 'loading matrix', n_jobs, col_label='motif') for f in loading_matrix_filenames]
    position = [{p: i for i, p in enumerate(t.rows)} for t in tables]
    for t, pos in zip(tables, position):
        if not any(p in pos for p in expr.rows):
            raise InputError(f'None of the {len(expr.rows)} promoter ids of the expression table "{expr.path}" '
                             f'(for example {_show(expr.rows, 3)}) occur in the loading matrix "{t.path}" (for '
                             f'example {_show(t.rows, 3)}).' + _no_overlap_hint(expr, t))
    common = [i for i, p in enumerate(expr.rows) if all(p in pos for pos in position)]
    if not common:
        raise InputError('No promoter is present in the expression table and in all loading matrices '
                         f'({", ".join(chr(34) + t.path + chr(34) for t in tables)}) at the same time.')
    for t, pos in zip(tables, position):
        absent = [p for p in expr.rows if p not in pos]
        if absent:
            warn(f'{len(absent)} of {len(expr.rows)} promoters of the expression table are absent from the '
                 f'loading matrix "{t.path}" and were dropped: {_show(absent)}.')
        expr_set = set(expr.rows)
        extra = [p for p in t.rows if p not in expr_set]
        if extra:
            warn(f'{len(extra)} of {len(t.rows)} promoters of the loading matrix "{t.path}" are absent from '
                 f'the expression table and were ignored: {_show(extra)}.')
    proms_common = [expr.rows[i] for i in common]
    Y_common = Y_all[common]
    del Y_all
    motif_names_all = [[f'{c}_{pf}' if pf is not None else c for c in t.cols] for t, pf in zip(tables, postfixes)]
    flat = [n for names in motif_names_all for n in names]
    if len(set(flat)) != len(flat):
        seen, dups = set(), list()
        for n in flat:
            if n in seen and n not in dups:
                dups.append(n)
            seen.add(n)
        raise InputError(f'Motif names are not unique across the loading matrices: {_show(dups)} occur(s) '
                         'in more than one matrix. Use --loading-postfix (one per loading matrix) to tell '
                         'them apart.')
    selected = [list(range(len(names))) for names in motif_names_all]
    if wanted_motifs is not None:
        known = set(flat)
        wanted = list()
        for tok in wanted_motifs:
            wanted.extend([x for x in tok.split(',') if x] if tok not in known and ',' in tok else [tok])
        wanted = list(dict.fromkeys(wanted))
        unmatched = [n for n in wanted if n not in known]
        wanted_set = set(wanted)
        selected = [[j for j, n in enumerate(names) if n in wanted_set] for names in motif_names_all]
        if not any(selected):
            raise InputError(f'None of the {len(wanted_motifs)} motif names in "{os.fspath(motif_names_filename)}" '
                             f'(for example {_show(wanted_motifs, 3)}) matches a motif of the loading matrices '
                             f'(for example {_show(flat, 3)}). Names are matched after --loading-postfix has '
                             'been appended to the motif names.')
        if unmatched:
            warn(f'{len(unmatched)} of {len(wanted)} motif names in "{os.fspath(motif_names_filename)}" '
                 f'match no motif and were ignored: {_show(unmatched)}.')

    # ---- motif expression (stored in the project, not used by the estimators) ----
    motif_expression_parts = None
    if motif_expression_filenames:
        motif_expression_parts = list()
        for f, names, pf, sel, t in zip(motif_expression_filenames, motif_names_all, postfixes, selected, tables):
            me = _read_table(f, 'motif expression table', n_jobs, col_label='sample')
            if pf is not None:
                me.rows = [f'{r}_{pf}' for r in me.rows]
            me_pos = {r: i for i, r in enumerate(me.rows)}
            lacking = [names[j] for j in sel if names[j] not in me_pos]
            if lacking:
                raise InputError(f'The motif expression table "{me.path}" has no row for {len(lacking)} motif(s) '
                                 f'of the loading matrix "{t.path}": {_show(lacking)}.')
            me_cols = {c: i for i, c in enumerate(me.cols)}
            if all(s in me_cols for s in sample_names):
                col_idx = [me_cols[s] for s in sample_names]
            elif groups_by_name and all(g in me_cols for g in groups_by_name):
                group_of = {s: g for g, members in groups_by_name.items() for s in members}
                col_idx = [me_cols[group_of[s]] for s in sample_names]
            else:
                lacking = [s for s in sample_names if s not in me_cols]
                raise InputError(f'The motif expression table "{me.path}" has no column for {len(lacking)} sample(s) of '
                                 f'the project: {_show(lacking)}. It needs one column per sample (or one per group).')
            K = _matrix(me, rows=[me_pos[names[j]] for j in sel], cols=col_idx, row_word='motif', col_word='sample')
            motif_expression_parts.append({names[j]: K[i] for i, j in enumerate(sel)})

    # ---- loadings of the promoters present in all the tables (checked before the filter is run) ----
    rows_common = [[pos[p] for p in proms_common] for pos in position]
    loadings_common = list()
    for k, (t, mode) in enumerate(zip(tables, loading_matrix_transformations)):
        if not selected[k]:
            loadings_common.append(None)
            continue
        # DRIST fits all motifs of a matrix jointly, the other transformations act on each column alone.
        sel = list(range(len(motif_names_all[k]))) if mode.startswith('drist') else selected[k]
        loadings_common.append(_matrix(t, rows=rows_common[k], cols=sel, row_word='promoter', col_word='motif'))
        t.frame = None
    del rows_common

    # ---- low-expression filter ----
    logger_print('Filtering promoters of low expression...', verbose)
    promoter_expression = pd.DataFrame(Y_common, index=proms_common, columns=sample_names)
    inds, weights = filter_lowexp(promoter_expression, cutoff=promoter_filter_lowexp_cutoff,
                                  fit_plot_filename=promoter_filter_plot_filename,
                                  max_mode=promoter_filter_max)
    n_kept = int(inds.sum())
    n_total = len(inds)
    logger_print(f'Low-expression filter: {n_total - n_kept} of {n_total} promoters were dropped, {n_kept} kept.', verbose)
    if n_kept == 0:
        raise InputError(f'The low-expression filter dropped all {n_total} promoters (w = {w}). Check that the '
                         'expression is on a log scale and raise --filter-lowexp-w.')
    if n_kept < 0.1 * n_total:
        warn(f'Only {n_kept} of {n_total} promoters ({n_kept / n_total:.0%}) passed the low-expression filter. '
             'Check that the expression is on a log scale, or raise --filter-lowexp-w.')
    promoter_expression = promoter_expression.loc[inds]
    proms = list(promoter_expression.index)
    kept = np.flatnonzero(inds)
    constant_samples = [s for s, sd in zip(sample_names, promoter_expression.to_numpy().std(axis=0)) if sd == 0]
    if len(constant_samples) == len(sample_names):
        raise InputError('The expression of every sample is constant across the promoters. Check the '
                         f'expression table "{expr.path}".')
    if constant_samples:
        warn(f'Sample(s) {_show(constant_samples)} have the same expression value for all promoters.')
    test_chromosomes  = list() # ['chr2', 'chr15']

    # ---- loadings ----
    blocks, motif_names = list(), list()
    for k, mode in enumerate(loading_matrix_transformations):
        if loadings_common[k] is None:
            continue
        names = motif_names_all[k]
        sel = list(range(len(names))) if mode.startswith('drist') else selected[k]
        X = loadings_common[k] if n_kept == n_total else loadings_common[k][kept]
        loadings_common[k] = None
        X, left = _transform_array(X, [names[j] for j in sel], mode, Y=promoter_expression,
                                   test_chromosomes=test_chromosomes, verbose=verbose)
        if mode.startswith('drist') and wanted_motifs is not None:
            chosen = {names[j] for j in selected[k]}
            keep = [j for j, n in enumerate(left) if n in chosen]
            X, left = X[:, keep], [left[j] for j in keep]
        if left:
            blocks.append(X)
            motif_names.extend(left)
        del X
    if not motif_names:
        raise InputError(f'No motif is left: after the loading transformation the loadings of all selected motifs '
                         f'are constant across the {n_kept} promoters that passed the low-expression filter '
                         '(see the warnings above for the reason). Check the loading matrices, the choice of '
                         '--loading-transform and --filter-lowexp-w.')
    loading_matrices = blocks[0] if len(blocks) == 1 else np.hstack(blocks)
    del blocks
    p, m = loading_matrices.shape
    s = len(sample_names)
    if s == 1:
        warn('The project has a single sample. "fit" needs at least two samples; only "maradoner bayes_linreg" '
             'can analyse this project.')
    if p <= m + 1:
        # Not an error: "fit --clustering" can still analyse such a project.
        warn(f'After filtering there are {p} promoters but {m} motifs, so there are more unknowns than '
             'promoters and "maradoner fit" cannot estimate the model for individual motifs. Use fewer motifs '
             '(--motif-filename), keep more promoters (a larger --filter-lowexp-w), provide more '
             'promoters, or group the motifs with "maradoner fit --clustering".')
    if motif_expression_parts is not None:
        lookup = {n: row for part in motif_expression_parts for n, row in part.items()}
        motif_expression = np.vstack([lookup[n] for n in motif_names])
    else:
        motif_expression = None
    proms = list(promoter_expression.index)
    promoter_expression = promoter_expression.values
    weights = np.asarray(weights)[inds]
    if sample_groups:
        index = {n: i for i, n in enumerate(sample_names)}
        sample_groups = {n: sorted(index[i] for i in members) for n, members in groups_by_name.items()}
    else:
        sample_groups = {n: [i] for i, n in enumerate(sample_names)}
    if covariates is not None:
        rank = np.linalg.matrix_rank(covariates)
        if len(sample_groups) - 1 > s - rank:
            warn(f'The covariates leave {s - rank} residual degrees of freedom (samples {s} - rank {rank} of the '
                 f'covariate design), but the {len(sample_groups)} groups need {len(sample_groups) - 1}: '
                 'group variances will not be identifiable. Use fewer covariates or fewer groups.')
    res = {'expression': promoter_expression,
           'loadings': loading_matrices,
           'motif_expression': motif_expression,
           'motif_postfixes': motif_postfixes,
           'promoter_names': proms,
           'sample_names': sample_names,
           'motif_names': motif_names,
           'weights': weights,
           'covariates': covariates,
           'covariate_names': covariate_names,
           'groups': sample_groups}
    if dump:
        logger_print('Saving project...', verbose)
        init_filename = stage_filename(project_name, 'init', compression)
        temporary = init_filename + '.tmp'
        try:
            with openers[compression](temporary, 'wb') as f:
                dill.dump(res, f)
        except BaseException:
            if os.path.isfile(temporary):
                os.remove(temporary)
            raise
        # Remove what belongs to a previous project of this name (exactly these files, nothing else).
        for stage in _PROJECT_STAGES:
            for fmt in openers:
                stale = stage_filename(project_name, stage, fmt)
                if stale != init_filename and os.path.isfile(stale):
                    os.remove(stale)
                    logger_print(f'Removed "{stale}" of the previous project.', verbose)
        os.replace(temporary, init_filename)
    return res
