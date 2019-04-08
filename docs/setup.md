# Setup Guide

This section is meant to help you with setting up an optimized environment for this course and assumes you are on Windows. If you have better alternatives for the programs below, feel free to use them or post them here - this case, please note that we will not discuss problems with setting up your own preferred environment during the meetings (troubleshooting would take forever) and simply assume you know what you are doing.

Download and setup the following infrastructure with default options:

1. [Pycharm Community 2019.1](https://www.jetbrains.com/pycharm/download/)
2. [Miniconda (latest)](https://docs.conda.io/en/latest/miniconda.html)
3. [Git Bash](https://gitforwindows.org/)

## Virtual Environments

`remember to enter your environment before you do anything on the terminal`

Miniconda is a robust package and environment manager. When you are working on multiple projects, you should keep your dependencies separate for each project, so you do not enter dependency hell. Practically, this means when you start your `Git Bash` console, the first thing for this course is to create a new environment (with `Python`) and then activate it (to enter the environment). You can always leave the enironment by pressing `Ctrl+A+D` or as follows:

```bash
conda create --name machine python=3.6  # create env
source activate machine                 # enter env
source deactivate machine               # exit env
```

Anaconda runs different channels where developers will upload their packages. One of the most useful channels is [`BioConda`](https://bioconda.github.io/) ([Nature Methods](https://doi.org/10.1038/s41592-018-0046-7)) for any program associated with biological data science and particularly genomic data science. Another useful channel is the `Forge` which is a community-driven developer's channel and usually has most up-to-date packages. We will add these to the configuration of `conda` on your system:

```bash
conda config --add channels defaults
conda config --add channels bioconda
conda config --add channels conda-forge
```

Anaconda also wraps around `pip` and attempts to manage both `conda` distributions and standard distributions by `PyPI`. We need a couple of Python packages for the first week: `scipy`, `pandas`, `numpy` and `matplotlib`, which we will install with Cnda

```bash
source activate machine
conda install scipy pandas numpy matplotlib
```

Great - all dependencies are now set up. Don't forget to exit your environment if you keep working on a different project!

**Linking environments to Pycharm**

If you are using Pycharm, you can set your interpreter (`python.exe`) to the one in the virtual environment - environments are basically just a separate directory structure that holds all the drivers, packages and dependency trees for the programs, and when you enter an environment, it is prepended to `$PATH` so that any program called from the command line will be looked for in the environmnet directory first.

In Pycharm, go to `
