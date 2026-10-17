
**MARADONER**

# MARADONER: Motif Activity Response Analysis Done Right

MARADONER is a tool for analyzing motif activities using promoter expression data. It provides a streamlined workflow to estimate parameters, predict deviations, and export results in a tabular form.

## Installation

MARADONER needs Python 3.11 or newer:

```bash
pip install maradoner
```

## Input files

All tables are plain text (.tsv or .csv, optionally gzipped) **with named rows and named columns**: the first line holds the column names and the first column holds the row names.

*   **Promoter expression** (`p` promoters x `s` samples), in log scale. Rows are promoters, columns are samples.
*   **Motif loadings** (`p` promoters x `m` motifs): how strongly each motif is associated with each promoter. Rows must use the same promoter names as the expression table; promoters that are missing from either table are dropped. Several loading matrices can be supplied at once (see `--loading-postfix`).
*   *(Optional)* **Sample groups**: which samples are replicates of the same condition. Either a JSON dictionary
    ```json
    {"control": ["sample_1", "sample_2"], "treated": ["sample_3", "sample_4"]}
    ```
    or a text file with one group per line: the group name followed by its sample names. Without it every sample is a group of its own.
*   *(Optional)* **Covariates** (`--covariates`): a table with sample names in the first column and one column per covariate (e.g. sex, age). Numeric covariates are z-scored, categorical ones are one-hot encoded.

## Basic Workflow


A typical MARADONER analysis session involves running commands sequentially for a given project:

1.  **`create`**: Initialize the project. This step parses your input files (promoter expression, motif loadings, optional motif expression, and sample groupings), performs initial filtering, and sets up the project's internal data structures.
    ```bash
    # Example: Initialize a project named 'my_project'
    maradoner create my_project path/to/expression.tsv path/to/loadings.tsv --sample-groups path/to/groups.json [other options...]
    ```
    *   Input files are typically tabular (.tsv, .csv), potentially compressed.
    *   You only need to provide input data files at this stage.
    *   Read the warnings: `create` reports promoters, samples and motifs that it had to drop.

2.  **`fit`**: Estimate the model's variance parameters and mean motif activities using the data prepared by `create`.
    ```bash
    maradoner fit my_project [options...]
    ```

3.  **`predict`**: Estimate the *deviations* of motif activities from their means for each sample or group, based on the parameters estimated by `fit`.
    ```bash
    maradoner predict my_project [options...]
    ```

4.  **`export`**: Save the final results, including estimated motif activities (mean + deviations), parameter estimates, goodness-of-fit statistics, and potentially statistical test results (like ANOVA) to a specified output folder.
    ```bash
    maradoner export my_project path/to/output_folder [options...]
    ```

The commands depend on each other: `fit` needs `create`, `predict` needs `fit`, and the commands that report results (`export`, `gof`, `anova`, `contrast`, `grn`) need `predict`. Re-running an earlier step invalidates the later ones: after a new `fit` run `predict` again (results of the outdated later steps are removed automatically).

### What `export` writes

| Path | Content |
| --- | --- |
| `README.md` | Generated description of every file in the folder and of its columns. `contrast`, `anova`, `grn`, `select-motifs`, `bayes_linreg` and `mara export` add their files to it when they write into the same folder. |
| `activities/activity_group_raw.tsv`, `activity_group_raw_std.tsv` | Predicted deviations of motif activities per group (motifs x groups) and their posterior standard deviations. |
| `activities/activity.tsv` | The two tables above divided by each other, i.e. standardized activities. |
| `activities/z_score.tsv` | Absolute z-scores per group and an `overall` score per motif. |
| `activities/activity_raw.tsv` | Predicted deviations per sample. |
| `params/motif_means.tsv` | Mean activity of each motif and its standard error (`params/covariates/` holds the same for every covariate, and `tests/covariates/` a joint test for each covariate of more than two levels). |
| `params/motif_variances.tsv`, `params/group_variances.tsv` | Variance parameters with standard errors: `tau` per motif; error variance `sigma` and activity variance `nu` per group. `filtered_motif_variances.tsv` lists the motifs that `predict` left out because their variance is negligible. |
| `params/promoter_means.tsv`, `params/sample_means.tsv` | Promoter-wise and sample-wise intercepts. |
| `params/correlations/` | Correlations between the parameter estimates. |
| `tests/asymptotics_based/sign.tsv` | Is the mean activity of a motif positive or negative? z statistic, one-sided p-values (`pvalue_pos`, `pvalue_neg`) and their FDR. |
| `tests/asymptotics_based/anova.tsv` | Does the activity of a motif vary across samples at all (`tau > 0`)? `stat`, `p-value`, `FDR` are a likelihood-ratio test; the `wald_*` columns hold the older, much less powerful Wald test. |
| `tests/prediction_based/anova.tsv` | Does the activity of a motif differ between groups? Based on the predicted deviations. |
| `params/family_variances.tsv`, `params/family_factors.tsv`, `tests/asymptotics_based/family_anova.tsv` | With `fit --motif-families`: the variance `omega` shared by the motifs of each family with its standard error, the factors of the covariance between families (`fit --motif-lowrank`) and a likelihood-ratio test of `omega = 0`. Without families the factors go to `params/motif_factors.tsv`. `params/activity_correlations.tsv` holds the correlations between motif activities that the fitted covariance implies. |
| `fov/` | Goodness-of-fit statistics, if `gof` was run. |

Motifs appear in the activity and test tables only if `predict` kept them (see `predict --filter-motifs`).

## Other Useful Commands

*   **`gof`**: After `predict`, calculate Goodness-of-Fit statistics (like Fraction of Variance Explained or Correlation) to evaluate how well the model components explain the observed expression data.
    ```bash
    maradoner gof my_project [options...]
    ```
    To evaluate the model on promoters it has not seen, hold some of them out at the `fit` stage and estimate their promoter-specific means before `gof`:
    ```bash
    maradoner fit my_project --test-chromosomes 3 --test-chromosomes X   # or --test-promoters-filename names.txt
    maradoner predict my_project
    maradoner estimate-promoter-mean my_project
    maradoner gof my_project
    ```
    The held-out promoters take no part in `fit`. Note, however, that the low-expression filter of `create` has already seen all promoters; use `create --filter-lowexp-w 1` if the testing set must not influence anything at all.
*   **`anova`**: Test whether motif activities differ between the listed groups.
    ```bash
    maradoner anova my_project anova.tsv control treated_6h treated_24h
    ```
*   **`contrast`**: Test a linear combination of group activities, written as an algebraic expression of group names.
    ```bash
    maradoner contrast my_project "treated - control" path/to/output_folder
    maradoner contrast my_project "(a + b + c) / 3 - d" path/to/output_folder
    ```
    With `--covariates`, the variables are covariates in place of groups, and their effects on motif activities are compared (`covariate_contrast.tsv`). The reference level of a categorical covariate stands for zero.
    ```bash
    maradoner contrast my_project "tissue_liver - tissue_muscle" path/to/output_folder --covariates
    ```
*   **`grn`**: Test each promoter against each motif in each group ("gene regulatory network"). The output is large: one promoters x motifs table per group.
    ```bash
    maradoner grn my_project path/to/output_folder
    ```
    In a project with covariates, `--covariate <name>` tests the effect of a covariate on motif activities in place of the activities. All samples are used, and one table per covariate is written to `belief/covariates`.
    ```bash
    maradoner grn my_project path/to/output_folder --covariate age
    ```
*   **`select-rank`**: Choose the number of factors for `fit --activity-lowrank`. Models with 1, 2, ... factors are fitted, and a factor is accepted if it lowers the REML deviance more than a factor does on data simulated from the model without it. Needs a fitted project and changes nothing in it.
    ```bash
    maradoner select-rank my_project --max-rank 3
    ```
*   **`select-motifs`**: If you provided multiple loading matrices in `create` (e.g., from different databases) with unique postfixes, this command helps select the single "best" variant for each motif based on statistical criteria. The output is a list of motif names intended to be used with the `--motif-filename` option in a subsequent `create` run.
    ```bash
    maradoner select-motifs my_project best_motifs.txt
    # Then, potentially re-run create using the generated list:
    # maradoner create my_project_filtered ... --motif-filename best_motifs.txt
    ```
*   **`bayes_linreg`**: A different, single-sample model for data without replication across samples (`fit` needs at least two samples). It goes straight from a project to result files.
    ```bash
    maradoner bayes_linreg my_project path/to/output_folder
    ```
*   **`generate`**: Create a synthetic dataset with known properties for testing or demonstration purposes.
    ```bash
    maradoner generate path/to/synthetic_data_output [options...]
    ```
*   **`mara`**: The classical (IS)MARA model, for comparison purposes (`maradoner mara --help`).

## Model extensions

*   `create --covariates covariates.tsv`: mean motif activities depend on sample covariates.
*   `create --covariates covariates.tsv --formula "age * sex + batch"`: a formula chooses columns of the table and adds their interactions (`a:b`; `a * b` stands for `a + b + a:b`).
*   `fit --error-lowrank k` / `fit --activity-lowrank k`: low-rank corrections for correlated noise / correlated activities across samples. `export` writes the factors to `params/error_factors.tsv` / `params/factors.tsv`; the latter is a valid table for `create --covariates`.
*   `fit --motif-families families.txt` / `fit --motif-lowrank k`: correlated activities of motifs. The covariance between motifs turns from the diagonal `Sigma` into `Sigma + F (Omega + L L^T) F^T`: `F` tells which motifs belong to which family, the diagonal `Omega` holds a variance per family, and the optional `L` of `k` columns is a covariance between families. The file of families is a JSON dictionary `{"family": ["motif", ...]}` or a text file with a family per line (`family motif motif ...`). Without families each motif forms a family of one, and the covariance is `Sigma + L L^T`.
*   `fit --motif-mean-prior ridge|families|covariance`: a prior on the mean activities of motifs, with its variances estimated by REML. It shrinks the means when the motifs are many for the number of promoters, and the levels of held-out promoters are predicted better.
*   `fit --clustering KMeans|NMF --num-clusters k`: replace motifs by clusters of motifs with similar loadings (useful when there are more motifs than the data can tell apart); all later commands then report clusters.
*   `fit --no-motif-variance`: one common variance for all motifs, closer to the classical (IS)MARA.
*   `fit --promoter-variance` *(experimental)*: heuristic reweighting of promoters by their noise level.

## Performance

MARADONER uses one thread for linear algebra by default: it mostly works with many small matrices, where the usual one-thread-per-core setting makes it considerably *slower*. Use `maradoner --threads N <command> ...` (or the `MARADONER_THREADS` environment variable) to change it; if `OMP_NUM_THREADS` and the like are already set in your environment they are respected.

On small and medium datasets a large part of the run time is spent compiling numerical code, the same code on every run. Set the environment variable `MARADONER_JAX_CACHE` to a folder (e.g. `export MARADONER_JAX_CACHE=~/.cache/maradoner`) to keep the compiled code between runs; repeated runs then start several times faster.

## When something goes wrong

Problems with the input (a malformed table, sample names that do not match, a step that was skipped) are reported as a short `Error: ...` message that says what to fix. A long traceback, on the other hand, means an unexpected internal error: please report it via the [issue tracker](https://github.com/autosome-ru/maradoner/issues). Set the environment variable `MARADONER_DEBUG=1` to get a traceback for every error.

## Getting Help

Each command has various options for customization. To see the full list of commands and their detailed options, use the `--help` flag:

```bash
maradoner --help
maradoner create --help
maradoner fit --help
# and so on for each command
```

There's also a project doumentation available at [maradoner.georgy.top](https://maradoner.georgy.top).

# Cite
Please cite the [conference paper](https://doi.org/10.1145/3822521.3822537) on MARADONER:

```
@inproceedings{meshcheryakov2026,
      title={MARADONER: Motif Activity Response Analysis Done Right},
      author={Meshcheryakov, Georgy and Buyan, Andrey and Favorov, Alexander and Makeev, Vsevolod Jurievich and Kulakovskiy, Ivan Vladimirovich},
      booktitle={Proceedings of the 2026 18th International Conference on Bioinformatics and Biomedical Technology},
      series={ICBBT 2026},
      pages={264--269},
      year={2026},
      month=may,
      publisher={ACM},
      location={Suzhou, China},
      doi={10.1145/3822521.3822537},
      url={https://doi.org/10.1145/3822521.3822537},
}
```

The [arXiv preprint](https://doi.org/10.48550/arXiv.2602.03343) is a longer manuscript with the mathematics in more detail, and it can be cited as well:

```
@misc{meshcheryakov2026arxiv,
      title={MARADONER: Motif Activity Response Analysis Done Right},
      author={Georgy Meshcheryakov and Andrey I. Buyan},
      year={2026},
      eprint={2602.03343},
      archivePrefix={arXiv},
      primaryClass={stat.CO},
      url={https://arxiv.org/abs/2602.03343},
}
```
