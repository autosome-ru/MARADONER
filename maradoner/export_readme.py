# -*- coding: utf-8 -*-
"""README.md of an exported results folder: one entry per file that is actually there."""
import json
import os
import re


# Options of the export that change the meaning of a column. They are stored in a comment
# at the end of README.md, so that a later command writing into the same folder
# (loadings-product, contrast) does not lose them.
_CONTEXT_TAG = '<!-- maradoner-export: '

_DEFAULTS = dict(project=None, version=None, n_motifs=None, n_filtered=0, by='group',
                 anova=None, zero_margin=None, weighted_zscore=False, fov_grouped=False,
                 counts_grouped=True, covariates=('intercept',), lp_intercepts=True,
                 # Set by the other commands that can write into a results folder.
                 fov_mara=False, grn_prior=0.5, grn_means=True, grn_fdr=0.05, grn_min_variance=1e-3,
                 # Files with user-chosen names: {relative path: [kind, details]}.
                 files=None)


def _activities(ctx):
    n = ctx['n_filtered']
    s = 'Deviations of motif activities from their means (the means are in `params/motif_means.tsv`).'
    if n:
        s += f' {n} motif(s) dropped by `predict` are absent, see `params/filtered_motif_variances.tsv`.'
    return s

def _overall(ctx):
    if ctx['weighted_zscore']:
        return '`overall`: absolute z-score of the inverse-variance weighted mean of the group activities.'
    return '`overall`: root mean square of the z-scores over groups.'

def _anova_asymptotic(ctx):
    mode = ctx['anova']
    if mode == 'lrt':
        cols = ['`stat`: likelihood-ratio statistic.',
                '`p-value`: from the 0.5 chi2(0) + 0.5 chi2(1) mixture.',
                '`FDR`: Benjamini-Hochberg over all motifs of the model, including the dropped ones.']
    elif mode == 'negative':
        cols = ['`stat`: Wald statistic tau / std.',
                '`p-value`: lower tail of the statistic. Small means tau is small relative to its '
                'standard error; it is not evidence that tau is zero.',
                '`FDR`: Benjamini-Hochberg.']
    else:
        cols = ['`stat`, `p-value`, `FDR`: one-sided Wald test (no likelihood-ratio statistics in this fit).']
    return cols + ['`wald_stat`, `wald_p-value`, `wald_fdr`: one-sided Wald test, tau / std. Low power.']

def _zero(ctx):
    margin = ctx['zero_margin']
    if margin is None:
        return ('`zero`, `fdr_zero`: P(|Z| <= |stat|). Small means the mean is close to zero relative to its '
                'standard error; it is not a p-value of a test that the mean is zero.')
    return f'`zero`, `fdr_zero`: equivalence test, H0: |mean| >= {margin:g}.'

def _fov_cols(ctx):
    if ctx['fov_mara']:
        return ['`stat`: statistic of the full model.']
    return ['`null`: promoter and sample means only.',
            '`means`: plus mean motif activities.',
            '`motif_means`: plus activity deviations, i.e. the full model.',
            'With `gof --stat-mode residual` the order is reversed: full model, then what remains '
            'after removing the means, then after removing the mean activities as well.']

def _fov_by(ctx):
    return 'groups' if ctx['fov_grouped'] else 'samples'

def _loadings_product(ctx):
    what = '(mean activity + deviation in the group)' if ctx['lp_intercepts'] else 'the activity deviation in the group'
    return f'One table per group: loading times {what}.'

def _grn_belief(ctx):
    what = 'mean activity + deviation' if ctx['grn_means'] else 'activity deviation'
    return (f'One table per group. Posterior probability that the motif ({what}) contributes to the expression '
            f'of the promoter in that group; prior probability {ctx["grn_prior"]:g}. Promoters with an error '
            f'variance below {ctx["grn_min_variance"]:g} are left out.')

def _grn_belief_cov(ctx):
    return ('One table per covariate (`maradoner grn --covariate`). Posterior probability that the effect of '
            'the covariate on the activity of the motif contributes to the expression of the promoter, over '
            f'all samples; prior probability {ctx["grn_prior"]:g}. Promoters with an error variance below '
            f'{ctx["grn_min_variance"]:g} are left out.')

def _grn_threshold(ctx):
    return (f'One number per group: promoter-motif pairs with belief >= it have a Bayesian FDR of '
            f'{ctx["grn_fdr"]:g}.')

def _registered(kind):
    def text(ctx, details=None):
        if kind == 'anova':
            return ('Does the activity of a motif differ between the groups ' +
                    ', '.join(f'`{g}`' for g in details or []) + '? (`maradoner anova`)')
        return ('Best variant of each motif, one name per line (`maradoner select-motifs`). '
                'Input of `create --motif-filename`.')
    return text

def _counts_by(ctx):
    return 'groups' if ctx['counts_grouped'] else 'samples'


# (pattern of the path relative to the results folder, shape, description, columns).
# Any of the last three may be a function of the context. Order is the order in README.
_FILES = [
    (r'B\.tsv', 'promoters x motifs', 'Loading matrix as stored in the project.', None),

    (r'activities/activity_group_raw\.tsv', 'motifs x groups',
     'Predicted activity deviation of each motif in each group.', None),
    (r'activities/activity_group_raw_std\.tsv', 'motifs x groups',
     'Posterior standard deviations of `activity_group_raw.tsv`.', None),
    (r'activities/activity\.tsv', 'motifs x groups',
     '`activity_group_raw.tsv` divided by `activity_group_raw_std.tsv`: signed z-scores.', None),
    (r'activities/z_score\.tsv', 'motifs x (1 + groups)',
     'Absolute values of `activity.tsv`.', lambda ctx: [_overall(ctx)]),
    (r'activities/activity_raw\.tsv', 'motifs x samples',
     'Activity deviation predicted from each sample alone. No standard deviations; equals '
     '`activity_group_raw.tsv` for groups of one sample.', None),

    (r'params/motif_means\.tsv', 'motifs',
     'Mean activity of each motif.', ['`mean`: estimate.', '`std`: standard error.']),
    (r'params/covariates/intercept\.tsv', 'motifs',
     'Mean motif activity with a test against zero.',
     ['`effect`, `std`: as in `motif_means.tsv`.', '`z`: effect / std.', '`pvalue`: two-sided.',
      '`fdr`: Benjamini-Hochberg.']),
    (r'params/covariates/[^/]+\.tsv', 'motifs',
     'Change of motif activity per unit of the covariate (numeric covariates are z-scored, '
     'categorical ones are one-hot encoded).',
     ['`effect`: estimate.', '`std`: standard error.', '`z`: effect / std.', '`pvalue`: two-sided.',
      '`fdr`: Benjamini-Hochberg.']),
    (r'params/motif_variances\.tsv', 'motifs',
     'Variance of activity deviations per motif. The variance of motif m in group g is tau_m * nu_g.',
     ['`tau`: estimate.', '`std`: standard error.']),
    (r'params/filtered_motif_variances\.tsv', 'motifs',
     'Motifs dropped by `predict` because their variance is negligible.', ['`tau`: estimate.']),
    (r'params/group_variances\.tsv', 'groups',
     'Variance parameters per group.',
     ['`sigma`, `sigma_std`: error variance and its standard error.',
      '`nu`, `nu_std`: activity variance multiplier and its standard error. `nu_std` is empty for '
      'the group whose `nu` is fixed for identifiability.']),
    (r'params/promoter_means\.tsv', 'promoters', 'Promoter-wise intercepts.', ['`mean`: estimate.']),
    (r'params/promoter\.tsv', 'promoters', 'Promoter-wise intercepts and weights (`fit --promoter-variance`).',
     ['`mean`: intercept.', '`weight`: weight of the squared residuals of the promoter in the fit.']),
    (r'params/promoter_variances\.tsv', 'promoters x groups', 'Promoter-wise variances.', None),
    (r'params/test_promoter_means\.tsv', 'held-out promoters',
     'Intercepts of the held-out promoters (`estimate-promoter-mean`).', ['`mean`: estimate.']),
    (r'params/sample_means\.tsv', 'samples', 'Sample-wise intercepts.', ['`mean`: estimate.']),
    (r'params/motif_context\.tsv', 'motifs', 'Loading context parameters.', None),
    (r'params/correlations/motif_means\.tsv', 'motifs x motifs',
     'Correlations between the estimates of motif means.', None),
    (r'params/correlations/motif_variances\.tsv', 'motifs x motifs',
     'Correlations between the estimates of tau.', None),
    (r'params/correlations/motif_group_variances\.tsv', 'groups x groups',
     'Correlations between the estimates of nu.', None),
    (r'params/correlations/error_variances\.tsv', 'groups x groups',
     'Correlations between the estimates of sigma.', None),
    (r'params/correlations/sample_means\.tsv', 'samples',
     'Same values as `params/sample_means.tsv`, without sample names. Kept for backward compatibility.', None),

    (r'tests/asymptotics_based/sign\.tsv', 'motifs',
     'Is the mean activity of a motif positive or negative?',
     lambda ctx: ['`stat`: mean / std.',
                  '`pvalue_neg`, `fdr_neg`: one-sided test, H1: mean < 0.',
                  '`pvalue_pos`, `fdr_pos`: one-sided test, H1: mean > 0.',
                  _zero(ctx)]),
    (r'tests/asymptotics_based/anova\.tsv', 'motifs',
     'Does the activity of a motif vary across samples at all? H0: tau = 0.', _anova_asymptotic),
    (r'tests/prediction_based/anova\.tsv', 'motifs',
     'Does the activity of a motif differ between groups? Chi-squared test on the predicted group '
     'activities and their posterior covariances.',
     ['`stat`: test statistic.', '`p-value`: chi2 with (groups - 1) degrees of freedom.',
      '`FDR`: Benjamini-Yekutieli.']),
    (r'(.*/)?contrast[^/]*\.tsv', 'motifs',
     'Test of a linear combination of group activities (`maradoner contrast`).',
     ['`z_stat`: contrast / its standard deviation.', '`pval`: two-sided.', '`fdr`: Benjamini-Yekutieli.']),

    (r'fov/(train|test)/total\.tsv', '1 row',
     'Goodness of fit over all data: fraction of variance explained, or Pearson correlation with '
     '`gof --stat-type corr`.', _fov_cols),
    (r'fov/(train|test)/promoters\.tsv', 'promoters', 'The same statistics per promoter.', None),
    (r'fov/(train|test)/samples\.tsv', _fov_by, 'The same statistics per column of the data.', None),

    (r'log_counts/(train|test)\.tsv', lambda ctx: f'promoters x {_counts_by(ctx)}',
     'Expression predicted by the full model.', None),
    (r'belief/[^/]+\.tsv', 'promoters x motifs', _grn_belief, None),
    (r'belief/[^/]+\.hdf', 'promoters x motifs', lambda ctx: _grn_belief(ctx) + ' HDF, float16, key `belief`.', None),
    (r'belief/[^/]+\.txt', '1 number', _grn_threshold, None),
    (r'belief/covariates/[^/]+\.tsv', 'promoters x motifs', _grn_belief_cov, None),
    (r'belief/covariates/[^/]+\.hdf', 'promoters x motifs',
     lambda ctx: _grn_belief_cov(ctx) + ' HDF, float16, key `belief`.', None),
    (r'belief/covariates/[^/]+\.txt', '1 number',
     lambda ctx: _grn_threshold(ctx).replace('per group', 'per covariate'), None),
    (r'lr/[^/]+\.tsv', 'promoters x motifs',
     'One table per group. Likelihood ratio of the full model against the model without the motif '
     'at that promoter, capped at exp(88.72).', None),
    (r'lr/[^/]+\.hdf', 'promoters x motifs',
     'One table per group. Likelihood ratio of the full model against the model without the motif '
     'at that promoter, capped at exp(88.72). HDF, float32, key `lrt`.', None),

    (r'lr/covariates/[^/]+\.tsv', 'promoters x motifs',
     'One table per covariate. Likelihood ratio of the full model against the model without the effect '
     'of the covariate on the motif at that promoter, capped at exp(88.72).', None),
    (r'lr/covariates/[^/]+\.hdf', 'promoters x motifs',
     'One table per covariate. Likelihood ratio of the full model against the model without the effect '
     'of the covariate on the motif at that promoter, capped at exp(88.72). HDF, float32, key `lrt`.', None),

    (r'motifs\.tsv', 'motifs, repeated per sample if there are several',
     'Single-sample Bayesian regression (`maradoner bayes_linreg`), sorted by |z| within a sample.',
     ['`sample`: present if the project has more than one sample.',
      '`activity`: posterior mean of the motif activity.',
      '`std`: posterior standard deviation given the estimated variances. Too small as an error bar '
      'of a non-zero activity.',
      '`z`: activity / std.', '`p_value`: two-sided.', '`fdr_bh`: Benjamini-Hochberg within the sample.',
      '`variance`, `variance_std`: prior variance of the activity and its standard error.',
      '`fov`: fraction of variance lost when the motif alone is removed.']),
    (r'samples\.tsv', 'samples',
     'Per-sample summary of `maradoner bayes_linreg`.',
     ['`mu`, `mu_std`: intercept and its standard error.',
      '`sigma2`, `sigma2_std`: error variance and its standard error.',
      '`fov`: fraction of variance explained.',
      '`reml_loglik`: restricted log-likelihood at the optimum.',
      '`grad_norm`: gradient norm at the optimum; should be near zero.',
      '`n_nonzero_variance`: motifs with a non-zero variance.',
      '`n_significant`: motifs below the FDR level.']),
    (r'promoters\.tsv', 'promoters x samples', 'Expression fitted by `maradoner bayes_linreg`.', None),

    (r'params\.tsv', 'samples', 'Variance parameters of the classical MARA model (`maradoner mara`).',
     ['`sigma`: error variance.', '`tau`: variance of motif activities.']),
    (r'activities_raw\.tsv', 'motifs x samples', 'MARA motif activities.', None),
    (r'activities\.tsv', 'motifs x (1 + samples)',
     '`activities_raw.tsv` divided by posterior standard deviations: signed z-scores.',
     ['`overall`: mean of the z-scores over samples.']),
    (r'z_scores\.tsv', 'motifs x (1 + samples)', 'Absolute values of `activities.tsv`.',
     ['`overall`: root mean square of the z-scores over samples.']),

    (r'loadings-product/[^/]+\.tsv', 'promoters x motifs', lambda ctx: _loadings_product(ctx), None),
    (r'loadings-product/[^/]+\.hdf', 'promoters x motifs',
     lambda ctx: _loadings_product(ctx) + ' HDF, float16, key `lrt`.', None),
]

_SECTIONS = {'': None,
             'activities': _activities,
             'params': None,
             'tests': 'Tables hold one row per motif; motifs dropped by `predict` appear only in `sign.tsv`.',
             'fov': 'Written if `gof` was run before the export.',
             'log_counts': None,
             'loadings-product': None,
             'belief': 'Written by `maradoner grn`.',
             'lr': 'Written by `maradoner grn --stat`.'}


def _resolve(x, ctx):
    return x(ctx) if callable(x) else x

def _describe(relpath: str):
    for item in _FILES:
        if re.fullmatch(item[0], relpath):
            return item
    return None

def _collapse(relpath: str, pattern: str) -> str:
    # Files that come one per group / covariate share an entry.
    for folder in ('belief', 'lr'):
        if pattern.startswith(folder + '/covariates/'):
            return f'{folder}/covariates/<covariate>' + os.path.splitext(relpath)[1]
    for folder in ('loadings-product', 'belief', 'lr'):
        if pattern.startswith(folder + '/'):
            return f'{folder}/<group>' + os.path.splitext(relpath)[1]
    if pattern == r'params/covariates/[^/]+\.tsv':
        return 'params/covariates/<covariate>.tsv'
    return relpath


_REGISTERED = {
    'anova': ('motifs', _registered('anova'),
              ['`stat`: test statistic.', '`p-value`: chi2 with (groups - 1) degrees of freedom.',
               '`FDR`: Benjamini-Yekutieli.']),
    'select': ('text', _registered('select'), None),
}
_OTHER_LIMIT = 30


def build_readme(output_folder: str, **context) -> str:
    ctx = dict(_DEFAULTS)
    ctx.update({k: v for k, v in context.items() if k in _DEFAULTS})
    registered = {k: v for k, v in (ctx['files'] or dict()).items()
                  if os.path.isfile(os.path.join(output_folder, k)) and v[0] in _REGISTERED}
    ctx['files'] = registered
    found = dict()
    other = list()
    for root, _, files in os.walk(output_folder):
        for name in files:
            rel = os.path.relpath(os.path.join(root, name), output_folder).replace(os.sep, '/')
            if rel in registered or rel == 'README.md':
                continue
            item = _describe(rel)
            if item is not None:
                found.setdefault(_collapse(rel, item[0]), item)
            else:
                other.append(rel)
    # Tables of the training and of the held-out promoters share an entry.
    for rel in [r for r in found if r.startswith(('fov/train/', 'log_counts/train'))]:
        twin = rel.replace('train', 'test', 1)
        if twin in found:
            _, shape, text, columns = found.pop(twin)
            item = found.pop(rel)
            text_both = lambda ctx, text=text: _resolve(text, ctx) + ' `test`: held-out promoters.'
            found[rel.replace('train', '{train,test}', 1)] = (item[0], shape, text_both, columns)
    order = {item[0]: i for i, item in enumerate(_FILES)}
    entries = sorted(found.items(), key=lambda kv: (order[kv[1][0]], kv[0]))
    # Root-level tables go first, so that they do not end up under the heading of a subfolder.
    entries = [e for e in entries if '/' not in e[0]] + [e for e in entries if '/' in e[0]]

    lines = ['# MARADONER results', '']
    head = list()
    if ctx['project']:
        head.append(f'Project `{ctx["project"]}`')
    if ctx['version']:
        head.append(f'MARADONER {ctx["version"]}')
    if head:
        lines += [', '.join(head) + '.', '']
    lines += ['Tab-separated tables unless stated otherwise; the first column holds row names.', '']
    if os.path.isdir(os.path.join(output_folder, 'activities')):
        lines += ['Activity of a motif in a group = its mean + its deviation in that group. '
                  'Standard errors and tests are asymptotic.', '']

    def entry(rel, shape, text, columns):
        lines.append(f'- `{rel}` ({shape}): {text}')
        for c in columns or []:
            lines.append(f'  - {c}')

    for rel, (kind, details) in sorted(registered.items()):
        shape, text, columns = _REGISTERED[kind]
        entry(rel, shape, text(ctx, details), columns)
    section = ''
    for rel, (_, shape, text, columns) in entries:
        top = rel.split('/')[0] if '/' in rel else ''
        if top != section:
            section = top
            lines += ['', f'## {top}/', '']
            note = _resolve(_SECTIONS.get(top), ctx)
            if note:
                lines += [note, '']
        entry(rel, _resolve(shape, ctx), _resolve(text, ctx), _resolve(columns, ctx))
    if other:
        other.sort()
        lines += ['', '## Other files', '', 'Not described here: not written by MARADONER, or left by an '
                  'older version.', '']
        lines += [f'- `{rel}`' for rel in other[:_OTHER_LIMIT]]
        if len(other) > _OTHER_LIMIT:
            lines.append(f'- ... and {len(other) - _OTHER_LIMIT} more')
    return '\n'.join(lines) + '\n\n' + _CONTEXT_TAG + json.dumps(ctx) + ' -->\n'


def _stored_context(filename: str) -> dict:
    try:
        with open(filename) as f:
            last = f.read().rstrip().rsplit('\n', 1)[-1]
        if last.startswith(_CONTEXT_TAG):
            return json.loads(last[len(_CONTEXT_TAG):-len(' -->')])
    except (OSError, ValueError):
        pass
    return dict()

def is_results_folder(folder: str) -> bool:
    """True if `folder` holds a README.md written by this module."""
    filename = os.path.join(folder or '.', 'README.md')
    try:
        with open(filename) as f:
            return _CONTEXT_TAG in f.read()
    except OSError:
        return False


def write_readme(output_folder: str, register: dict = None, **context):
    """(Re)write README.md of a results folder. Context of the previous README of the
    folder is kept, so a later command only has to pass what it adds. `register` maps
    files with user-chosen names (relative to the folder) to [kind, details].
    A README.md that MARADONER did not write is never overwritten."""
    filename = os.path.join(output_folder, 'README.md')
    if os.path.isfile(filename) and not is_results_folder(output_folder):
        from .utils import warn
        warn(f'"{filename}" was not written by MARADONER and is left as it is: no description '
             'of the result files is written.')
        return
    ctx = _stored_context(filename)
    ctx.update(context)
    if register:
        files = dict(ctx.get('files') or dict())
        files.update(register)
        ctx['files'] = files
    with open(filename, 'w') as f:
        f.write(build_readme(output_folder, **ctx))


def register_file(filename: str, kind: str, details=None):
    """Describe a file with a user-chosen name, if it was put into a results folder (one
    that already has a README.md of ours). No README.md is created in other folders."""
    folder, name = os.path.split(os.fspath(filename))
    folder = folder or '.'
    if is_results_folder(folder):
        write_readme(folder, register={name: [kind, details]})
