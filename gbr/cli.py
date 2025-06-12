import argparse
import typing
import typing as t

import scanpy as sc
import yaml
#
from devtools import pprint
from pydantic import BaseModel, Field

from . import (ARBRunner, GBRunner, GBRArgs, GBMethod, XGBArgs,
               gbrunner_args)
from .data import (CSVDataArgs, CSVDataProcessor, ExpDataProcessor, SCDataArgs,
                   SCDataProcessor)

#
sc.settings.verbosity = 3
sc.settings.set_figure_params(dpi=80, facecolor="white")
sc.logging.print_header()


class InputArgs(BaseModel):
    format: t.Literal["ad", "csv"] = "ad"
    method: GBMethod = "xgb"
    device: t.Literal["cpu", "gpu"] = "cpu"
    in_data: SCDataArgs | CSVDataArgs = Field(discriminator="format")
    regressor_args: GBRArgs = Field(discriminator="regressor", default=XGBArgs())
    take_n: int | None = 20
    use_tqdm: bool = True
    rstats_file: str = "./run_stats.json"
    network_file: str = "./network.csv"
    scdata_save_file: str | None = ""


def run_grad_boost(rargs: InputArgs, exp_data: ExpDataProcessor):
    match rargs.method:
        case "arb:default" | "arb:gbm":
            arbr = ARBRunner(exp_data)
            arbr.build(rargs.method)
            arbr.dump(rargs.rstats_file, rargs.network_file)
        case "xgb" | "stxgb" | "sgbr" | "stsgbr" | "lgb" | "stlgb":
            xgbr = GBRunner(exp_data, rargs.device)
            xgbr.build(
                rargs.method, rargs.regressor_args,
                take_n=rargs.take_n, use_tqdm=rargs.use_tqdm
            )
            xgbr.dump(rargs.rstats_file, rargs.network_file)


def gen_scd_network(rargs: InputArgs):
    print("Run Arguments H5AD :: ", rargs)
    sc_data = None
    in_data: SCDataArgs = t.cast(SCDataArgs, rargs.in_data)
    match in_data.data:
        case "dense":
            in_data.scale_regress = True
            sc_data = SCDataProcessor(in_data)
        case "sparse":
            in_data.scale_regress = False
            sc_data = SCDataProcessor(in_data)
        case _:  # pyright: ignore[reportUnnecessaryComparison]
            sc_data = None  # pyright: ignore[reportUnreachable]
    if sc_data is None:
        return
    if rargs.scdata_save_file:
        sc_data.save(rargs.scdata_save_file)
    sc_data.print()
    run_grad_boost(rargs, sc_data)


def gen_csv_network(rargs: InputArgs):
    in_data: CSVDataArgs = t.cast(CSVDataArgs, rargs.in_data)
    print("Run Arguments CSV :: ", in_data)
    cv_data = CSVDataProcessor(in_data)
    cv_data.print()
    run_grad_boost(rargs, cv_data)


def gen_network(rargs: InputArgs):
    match rargs.format:
        case "ad":
            gen_scd_network(rargs)
        case "csv":
            gen_csv_network(rargs)


def data_args(cmdargs: argparse.Namespace):
    match cmdargs.format:
        case "csv":
            return CSVDataArgs(csv_file=cmdargs.csv_file)
        case "ad":
            return SCDataArgs(
                data=cmdargs.data,
                tf_file=cmdargs.tf_file,
                h5ad_file=cmdargs.h5ad_file,
                select_hvg=cmdargs.select_hvg,
                ntop_genes=cmdargs.ntop_genes,
                nsub_cells=cmdargs.nsub_cells,
            )
        case _:
            return None


def cmd_args(cmdargs: argparse.Namespace):
    in_data = data_args(cmdargs)
    if not in_data:
        return
    return InputArgs(
        format=cmdargs.format,
        method=cmdargs.method,
        device=cmdargs.device,
        in_data=in_data,
        regressor_args=gbrunner_args(cmdargs.method),
        take_n=cmdargs.take_n if cmdargs.take_n > 0 else None,
        use_tqdm=cmdargs.use_tqdm,
        rstats_file=cmdargs.rstats_out_file,
        network_file=cmdargs.out_file,
    )


def parse_yaml(yaml_file: str):
    with open(yaml_file) as ymfx:
        return yaml.safe_load(ymfx)


def yaml_args(cmdargs: argparse.Namespace):
    cfg_dict = parse_yaml(cmdargs.yaml_file)
    run_args = InputArgs.model_validate(cfg_dict)
    if run_args.method != run_args.regressor_args.regressor:
        run_args.regressor_args = gbrunner_args(run_args.method)
    # if cmdargs.take_n > 0:
    #     run_args.take_n=cmdargs.take_n
    # run_args.use_tqdm=cmdargs.use_tqdm
    # run_args.rstats_file=cmdargs.rstats_out_file
    # run_args.network_file=cmdargs.out_file
    return run_args


def input_args(cmdargs: argparse.Namespace):
    if cmdargs.format == "yaml":
        return yaml_args(cmdargs)
    return cmd_args(cmdargs)


def main(cmdargs: argparse.Namespace):
    run_args = input_args(cmdargs)
    print("Parsed Arguments :: ")
    pprint(run_args)
    print("--------------------")
    if run_args:
        gen_network(run_args)


if __name__ == "__main__":
    sc_data = SCDataArgs()
    def_args = InputArgs(in_data=sc_data)
    parser = argparse.ArgumentParser(
        prog="grb.cli", description="Generate GRNs w. XGB/lightgbm for Single Cell Data"
    )
    #
    subparsers = parser.add_subparsers(
        dest="format",
        required=True,
        help="CSV file / AnnData H5AD file / YAML with the data locations",
    )
    csv_parser = subparsers.add_parser("csv", help="CSV File as Input")
    csv_parser.add_argument("--csv_file", required=True)
    csv_parser.add_argument(
        "--device", "-c", choices=["cpu", "gpu"], default=def_args.device
    )
    csv_parser.add_argument(
        "--method",
        "-m",
        choices=typing.get_args(GBMethod),
        default=def_args.method,
    )
    csv_parser.add_argument("--rstats_out_file", default=def_args.rstats_file)
    csv_parser.add_argument("--out_file", default=def_args.network_file)
    #
    ad_parser = subparsers.add_parser("ad", help="H5AD File as Input")
    ad_parser.add_argument("--ntop_genes", type=int, default=sc_data.ntop_genes)
    ad_parser.add_argument("--data", "-d", choices=["dense", "sparse"], default="dense")
    ad_parser.add_argument("--nsub_cells", type=int, default=sc_data.nsub_cells)
    ad_parser.add_argument("--tf_file", default=sc_data.tf_file)
    ad_parser.add_argument("--h5ad_file", default=sc_data.h5ad_file)
    ad_parser.add_argument("--select_hvg", default=sc_data.select_hvg)
    ad_parser.add_argument("--take_n", default=0)
    ad_parser.add_argument(
        "--device", "-c", choices=["cpu", "gpu"], default=def_args.device
    )
    ad_parser.add_argument(
        "--method",
        "-m",
        choices=typing.get_args(GBMethod),
        default=def_args.method,
    )
    ad_parser.add_argument("--use_tqdm", action="store_true")
    ad_parser.add_argument("--rstats_out_file", default=def_args.rstats_file)
    ad_parser.add_argument("--out_file", default=def_args.network_file)
    #
    yaml_parser = subparsers.add_parser("yaml", help="YAML File as Input")
    yaml_parser.add_argument(
        "yaml_file", help="Yaml Input file with Data Configuration"
    )
    run_args = parser.parse_args()
    print("--------------------")
    print("Command line Arguments :: ")
    pprint(run_args)
    print("--------------------")
    main(run_args)
    print("--------------------")
