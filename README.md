This repository contains the code to implement the methods and simulations described in ["Estimating Counterfactual Matrix Means with Short Panel Data"](https://arxiv.org/abs/2312.07520) by Lihua Lei and Brad Ross

## Setup

The code in this repository is written in a mix of `julia` and `R`, so both need to be installed to be able to run it.

Unfortunately, we have not yet put together a setup script that automatically installs all required `julia` and `R` packages, so for now, make sure to install the packages listed at the top of each file before running anything. 

The scripts that execute the empirical application based on the Veneto Worker Histories (VWH) data (see the next section for a list) require the creation of `data` and `output` directories in the root directory of this repository, as well as `data/raw_data` and `data/clean_data` subdirectories. Before running any of the scripts listed below, you must also set the `DATA_PATH` and `OUTPUT_PATH` environment variables to the paths to your `data` and `output` directories, respectively. The raw Veneto Worker Histories `.dta` files obtained via the instructions [here](https://www.frdb.org/en/dati/dati-inps-carriere-lavorative-in-veneto/) must also be downloaded and placed in the `data/raw_data` directory.

## Scripts for the Empirical Application

*Note: the scripts below were run on a server with 190 GB of RAM, which allowed for a maximum of 20 cores to be used for multithreading. Depending on the computer on which the code is being run, the number of cores might need to be tuned to avoid running out of memory, which in turn could affect running time.*

1. `code/raw_vwh_dta_to_parquet.R`
2. `code/cluster_vwh_firms.jl`
3. `code/simulations_vwh.jl`
4. `code/visualize_panels.R`
5. `code/visualize_sim_results_vwh.R`

## Documentation for API

Unfortunately, we have also not had time to write proper documentation for the generic code that can be used to implement our estimation procedure across settings; please refer to `code/simulations_vwh.jl` to see how to use our code.