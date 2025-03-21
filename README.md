# GRN with Gradient Boosting

Utilities and runner scripts for generating gene regulatory networks with arboreto, XGBoost and Light GBM.

## Usage

Runner can be used to run either XG Boost or light GBM using the AnnData h5ad
file or CSV file as input

### Command-line parameters to select method, statistics and output files

Options to provide device, method, statistics output file and network output
file are mandatory irrespective of wether the input is a AnnData h5ad file or
CSV file. The options are provided as follows:

    python -m grb.cli
            [-h] [--take_n TAKE_N]
            [--device {cpu,gpu}]
            [--method {xgb,sgbr,arb:default,arb:gbm,lgb}]
            [--use_tqdm]
            [--rstats_out_file RSTATS_OUT_FILE]
            [--out_file OUT_FILE]
            {csv,ad} ...

    Generate GRNs w. XGB/lightgbm for Single Cell Data

    positional arguments:
      {csv,ad}              CSV or AnnData H5AD file
        csv                 CSV File as Input
        ad                  H5AD File as Input

    options:
      -h, --help            show this help message and exit
      --take_n TAKE_N
      --device, -c {cpu,gpu}
      --method, -m {xgb,sgbr,arb:default,arb:gbm,lgb}
      --use_tqdm
      --rstats_out_file RSTATS_OUT_FILE
      --out_file OUT_FILE

### Command-line parameters for csv file as input

Option to provide csv file as input is as follows:

     python -m grb.cli csv
            [-h]
            --csv_file CSV_FILE

     options:
       -h, --help           show this help message and exit
       --csv_file CSV_FILE

### Command-line parameters for anndata h5ad file as input

Option to provide AnnData h5ad file as input is as follows:

    python -m grb.cli ad
           [-h]
           [--ntop_genes NTOP_GENES]
           [--data {dense,sparse}]
           [--nsub_cells NSUB_CELLS]
           [--tf_file TF_FILE]
           [--h5ad_file H5AD_FILE]
           [--select_hvg SELECT_HVG]

    options:
      -h, --help            show this help message and exit
      --ntop_genes NTOP_GENES
      --data, -d {dense,sparse}
      --nsub_cells NSUB_CELLS
      --tf_file TF_FILE
      --h5ad_file H5AD_FILE
      --select_hvg SELECT_HVG
