"""Sequence features of promoters for "estimate-promoter-mean --covariate" (maradoner sequence-features).

The input is a FASTA of promoter sequences in the orientation of the promoter, all with the TSS
at the same position: for example "bedtools getfasta -s -name+" of windows from 1000 bp upstream
to 500 bp downstream of the TSS. The name of a promoter is the part of the record name before
"::". For each bin of positions relative to the TSS the table holds the counts of all k-mers,
the GC content and the CpG observed/expected ratio, one row per promoter.
"""
import itertools
import numpy as np
import pandas as pd
from .utils import InputError


def read_fasta(filename: str) -> dict:
    """Sequences by name, upper case. The name is the record name up to "::" or a space."""
    seqs, name = dict(), None
    with open(filename) as f:
        for line in f:
            if line.startswith('>'):
                name = line[1:].strip().split('::')[0].split()[0]
                if name in seqs:
                    raise InputError(f'The promoter "{name}" occurs more than once in "{filename}".')
                seqs[name] = list()
            elif name is not None:
                seqs[name].append(line.strip().upper())
    if not seqs:
        raise InputError(f'No sequence was found in the FASTA file "{filename}".')
    return {k: ''.join(v) for k, v in seqs.items()}


def sequence_features(seqs: dict, tss: int, bins: list, k: int = 5) -> pd.DataFrame:
    """k-mer counts, GC content and CpG o/e in the bins [bins[i], bins[i + 1]) of positions
    relative to the TSS (position tss of every sequence, 0-based). Positions outside a sequence
    are left out of its counts."""
    if k < 1 or k > 8:
        raise InputError(f'--k must be between 1 and 8, but {k} was given.')
    bins = [int(b) for b in bins]
    if len(bins) < 2 or any(a >= b for a, b in zip(bins, bins[1:])):
        raise InputError('--bins must hold at least 2 increasing positions relative to the TSS, '
                         f'but {bins} was given.')
    kmers = [''.join(p) for p in itertools.product('ACGT', repeat=k)]
    index = {s: i for i, s in enumerate(kmers)}
    n_k = len(kmers)
    names = list(seqs)
    out = np.zeros((len(names), (len(bins) - 1) * (n_k + 2)), dtype=np.float32)
    for r, name in enumerate(names):
        s = seqs[name]
        codes = np.array([index.get(s[i:i + k], -1) for i in range(len(s) - k + 1)], dtype=int)
        for b, (lo, hi) in enumerate(zip(bins, bins[1:])):
            a, z = max(tss + lo, 0), max(min(tss + hi, len(s)), 0)
            c = codes[a:max(z - k + 1, a)]
            o = b * (n_k + 2)
            out[r, o:o + n_k] = np.bincount(c[c >= 0], minlength=n_k)
            t = s[a:z]
            length = max(len(t), 1)
            n_c, n_g = t.count('C'), t.count('G')
            out[r, o + n_k] = (n_c + n_g) / length
            out[r, o + n_k + 1] = t.count('CG') * length / max(n_c * n_g, 1)
    cols = [f'{lo}..{hi}:{x}' for lo, hi in zip(bins, bins[1:]) for x in kmers + ['gc', 'cpg_oe']]
    return pd.DataFrame(out, index=pd.Index(names, name='promoter'), columns=cols)


def make_sequence_features(fasta: str, output: str, tss: int, bins: list, k: int = 5) -> pd.DataFrame:
    table = sequence_features(read_fasta(fasta), tss=tss, bins=bins, k=k)
    table.to_csv(output, sep='\t', float_format='%.6g')
    return table
