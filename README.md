
**MARADONER**

# MARADONER: Motif Activity Response Analysis Done Right

MARADONER is a tool for analyzing motif activities using promoter expression data. It provides a streamlined workflow to estimate parameters, predict deviations, and export results in a tabular form.

## Basic Workflow


A typical MARADONER analysis session involves running commands sequentially for a given project:

1.  **`create`**: Initialize the project. This step parses your input files (promoter expression, motif loadings, optional motif expression, and sample groupings), performs initial filtering, and sets up the project's internal data structures.
    ```bash
    # Example: Initialize a project named 'my_project'
    maradoner create my_project path/to/expression.tsv path/to/loadings.tsv --sample-groups path/to/groups.json [other options...]
    ```
    *   Input files are typically tabular (.tsv, .csv), potentially compressed.
    *   You only need to provide input data files at this stage.

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

## Other Useful Commands

*   **`gof`**: After `fit`, calculate Goodness-of-Fit statistics (like Fraction of Variance Explained or Correlation) to evaluate how well the model components explain the observed expression data.
    ```bash
    maradoner gof my_project [options...]
    ```
*   **`select-motifs`**: If you provided multiple loading matrices in `create` (e.g., from different databases) with unique postfixes, this command helps select the single "best" variant for each motif based on statistical criteria. The output is a list of motif names intended to be used with the `--motif-filename` option in a subsequent `create` run.
    ```bash
    maradoner select-motifs my_project best_motifs.txt
    # Then, potentially re-run create using the generated list:
    # maradoner create my_project_filtered ... --motif-filename best_motifs.txt
    ```
*   **`generate`**: Create a synthetic dataset with known properties for testing or demonstration purposes.
    ```bash
    maradoner generate path/to/synthetic_data_output [options...]
    ```

## Getting Help

Each command has various options for customization. To see the full list of commands and their detailed options, use the `--help` flag:

```bash
maradoner --help
maradoner create --help
maradoner fit --help
# and so on for each command