import argparse
import typing
import scanpy as sc
import dataclasses as dcl
from typing import Literal, TypeAlias
from .data import (CSVDataArgs, CSVDataProcessor, ExpDataProcessor,
                   SCDataProcessor, SCDataArgs)
from . import RegressorMethod, GBRunner, ARBRunner
#
sc.settings.verbosity = 3
sc.settings.set_figure_params(dpi=80, facecolor="white")
sc.logging.print_header()


@dcl.dataclass
class XGBArgs:
    learning_rate : float = 0.01
    n_estimators  : int = 500


@dcl.dataclass
class SGBMArgs: 
    learning_rate : float = 0.01
    n_estimators  : float = 500
    max_features  : float = 0.1


@dcl.dataclass
class LGBArgs:
    n_estimators: int = 500
    learning_rate: float = 0.01
    n_jobs: float = -1
    objective: str = 'regression'
    importance_type: str = 'gain'


RegressorArgs : TypeAlias = (
    XGBArgs |
    SGBMArgs |
    LGBArgs
)


@dcl.dataclass
class InputArgs:
    format: Literal['ad', 'csv'] = 'ad'
    method : RegressorMethod = 'xgb'
    device : Literal['cpu', 'gpu'] = 'cpu'
    scd : SCDataArgs | None = dcl.field(default_factory=SCDataArgs)
    csvd : CSVDataArgs | None = dcl.field(default_factory=CSVDataArgs)
    reg_args : RegressorArgs = dcl.field(default_factory=XGBArgs)
    take_n : int | None =20
    use_tqdm: bool = True
    rstats_file : str="./run_stats.json"
    network_file: str = "./network.csv"


def gb_args(method: RegressorMethod) -> RegressorArgs:
    match method:
        case 'sgbr':
            return SGBMArgs()
        case 'lgb':
            return LGBArgs()
        case _:
            return XGBArgs()


def run_grad_boost(rargs: InputArgs, exp_data: ExpDataProcessor):
    match rargs.method:
        case 'arb:default' | 'arb:gbm':
            arbr = ARBRunner(exp_data)
            arbr.build(rargs.method)
            arbr.dump(rargs.rstats_file, rargs.network_file)
        case 'xgb' | 'sgbr' | 'lgb':
            xgbr = GBRunner(exp_data)
            bargs = dcl.asdict(rargs.reg_args)
            if rargs.device == 'gpu':
                bargs["device"] = "cuda:0"
            xgbr.build(
                rargs.method,
                take_n=rargs.take_n,
                use_tqdm=rargs.use_tqdm,
                **bargs
            )
            xgbr.dump(rargs.rstats_file, rargs.network_file)


def gen_scd_network(rargs: InputArgs):
    print("Run Arguments", rargs)
    if rargs.scd is None:
        return
    sc_data = None
    match rargs.scd.data:
        case 'dense':
            rargs.scd.scale_regress = True
            sc_data = SCDataProcessor(rargs.scd)
        case 'sparse':
            rargs.scd.scale_regress = False
            sc_data = SCDataProcessor(rargs.scd)
        case _:  # pyright: ignore[reportUnnecessaryComparison]
            sc_data = None  # pyright: ignore[reportUnreachable]
    if sc_data is None:
        return 
    run_grad_boost(rargs, sc_data)


def gen_csv_network(rargs: InputArgs):
    print("Run Arguments CSV :: ", rargs.csvd)
    if rargs.csvd is None:
        return
    cv_data = CSVDataProcessor(rargs.csvd)
    print("CV Data", cv_data)
    run_grad_boost(rargs, cv_data)


def gen_network(rargs: InputArgs):
    match rargs.format:
        case 'ad':
            gen_scd_network(rargs)
        case 'csv':
            gen_csv_network(rargs)


def main(cmdargs: argparse.Namespace):
    csvd_args = (
        CSVDataArgs(csv_file=cmdargs.csv_file)
        if cmdargs.format == 'csv' else  None
    )
    scd_args = (
        SCDataArgs(
            data=cmdargs.data,
            tf_file=cmdargs.tf_file,
            h5ad_file=cmdargs.h5ad_file,
            select_hvg=cmdargs.select_hvg,
            ntop_genes=cmdargs.ntop_genes,
            nsub_cells=cmdargs.nsub_cells
        )
        if cmdargs.format == 'ad' else None
    )
    run_args = InputArgs(
        format=cmdargs.format,
        method=cmdargs.method, 
        device=cmdargs.device,
        scd=scd_args,
        csvd=csvd_args,
        reg_args=gb_args(cmdargs.method),
        take_n=cmdargs.take_n if cmdargs.take_n > 0 else None,
        use_tqdm=cmdargs.use_tqdm,
        rstats_file=cmdargs.rstats_out_file,
        network_file=cmdargs.out_file,
    )
    gen_network(run_args)


if __name__ == "__main__":
    def_args = InputArgs(scd=SCDataArgs())
    parser = argparse.ArgumentParser(
        prog="grb.cli",
        description="Generate GRNs w. XGB/lightgbm for Single Cell Data"
    )
    parser.add_argument('--take_n', default=0)
    parser.add_argument(
        '--device', '-c', choices=['cpu', 'gpu'], default='cpu'
    )
    parser.add_argument(
        '--method', '-m',
        choices=typing.get_args(RegressorMethod),
        default='xgb'
    )
    parser.add_argument('--use_tqdm', action='store_true')
    parser.add_argument('--rstats_out_file', default=def_args.rstats_file)
    parser.add_argument('--out_file', default=def_args.network_file)
    #
    subparsers = parser.add_subparsers(
        dest="format",
        required=True,
        help="CSV or AnnData H5AD file"
    )
    csv_parser = subparsers.add_parser('csv', help="CSV File as Input")
    csv_parser.add_argument('--csv_file', required=True)
    #
    ad_parser = subparsers.add_parser('ad', help="H5AD File as Input")
    ad_parser.add_argument(
        '--ntop_genes', type=int, default=def_args.scd.ntop_genes
    )
    ad_parser.add_argument(
        '--data', '-d', choices=['dense', 'sparse'], default='dense'
    )
    ad_parser.add_argument(
        '--nsub_cells', type=int, default=def_args.scd.nsub_cells
    )
    ad_parser.add_argument('--tf_file', default=def_args.scd.tf_file)
    ad_parser.add_argument('--h5ad_file', default=def_args.scd.h5ad_file)
    ad_parser.add_argument('--select_hvg', default=def_args.scd.select_hvg)
    #
    run_args = parser.parse_args()
    print(run_args)
    main(run_args)
