import itertools
import json
import pathlib
import dataclasses as dcl
import numpy as np
import polars as pl
import sklearn.metrics

XGB_GPU_SPARSE_NETWORKS = [
    {
        "dir": "./networks/",
        "stats": "xgb_gpu_stats.json",
        "adj": "xgb_gpu_sparse.csv",
    }
]

XGB_GPU_DENSE_NETWORKS = [
    {
        "dir": "./networks/",
        "stats": "xgb_gpu_dense_stats.json",
        "adj": "xgb_gpu_dense.csv",
    }
]

XGB_SPARSE_NETWORKS = [
    {
        "dir": "./networks/",
        "stats": "xgb_stats_sparse.json",
        "adj": "xgb_sparse.csv",
    }
]

XGB_DENSE_NETWORKS = [
    {
        "dir": "./networks/",
        "stats": "xgb_stats_dense.json",
        "adj": "xgb_dense.csv",
    }
]

XGB20K_GPU_DENSE_NETWORKS = [
    {
        "dir": "./networks/",
        "stats": "xgb20k_gpu_dense_stats.json",
        "adj": "xgb20k_gpu_dense.csv",
    }
]

LGB20K_DENSE_NETWORKS = [
    {
        "dir": "./networks/",
        "stats": "lgb20k_stats_dense.json",
        "adj": "lgb20k_dense.csv",
    }
]

LGB20K_SPARSE_NETWORKS = [
    {
        "dir": "./networks/",
        "stats": "lgb20k_stats_sparse.json",
        "adj": "lgb20k_sparse.csv",
    }
]

ARB20K_DEFAULT_NETWORKS = [
    {
        "dir": "./networks/",
        "stats": "arb20k_default_stats.json",
        "adj": "arb20k_default.csv",
    }
]


NETWORK_DATA_FILES = {
    "xgboost-gpu-sparse": XGB_GPU_SPARSE_NETWORKS,
    "xgboost-gpu-dense": XGB_GPU_DENSE_NETWORKS,
    "xgboost-sparse": XGB_SPARSE_NETWORKS,
    "xgboost-dense": XGB_DENSE_NETWORKS,
}

NETWORK20K_DATA_FILES = {
    "arboreto20k-default": ARB20K_DEFAULT_NETWORKS,
    "xgboost20k-gpu-dense": XGB20K_GPU_DENSE_NETWORKS,
    "lightgbm20k-dense": LGB20K_DENSE_NETWORKS,
}



#
TRRUST_COLUMNS = ["TF", "TARGET", "DIR", "REF"]
PBMC_BLOOD_COLUMNS = ["TF", "TARGET"]
PBMC_COLUMNS = ["TF", "TARGET"]
PBMC_BLOOD_TRRUST_COLUMNS = ["TF", "TARGET"]
PBMC_TRRUST_COLUMNS = ["TF", "TARGET", "DIR", "REF"]
REGNET_COLUMNS = ["TF", "TARGET", "WT"]
#
PBMC_BLOOD_NET_FILE = pathlib.Path("./gtruth/", "PBMC-Blood.csv")
PBMC_NET_FILE = pathlib.Path("./gtruth/", "PBMC.csv")
PBMC_TRRUST_NET_FILE = pathlib.Path("./gtruth/", "PBMC-TRRUST.csv")
PBMC_BLOOD_TRRUST_NET_FILE = pathlib.Path("./gtruth/", "PBMC-Blood-TRRUST.csv")


@dcl.dataclass
class AUCArgs:
    ntop_edges: int = 0
    ntop_regulons: int = 0
    subset_tfs: bool = False
    subset_tgts: bool = False
    threshold: float = 0.0
    restrict_tf_set: set[str] = dcl.field(default_factory=set)
    restrict_tgt_set: set[str] = dcl.field(default_factory=set)


class TFTGTNet:
    def __init__(self, tf_set: set[str], tgt_set:set[str], edge_set) -> None:
        self.tf_set : set[str]= tf_set
        self.tgt_set : set[str]= tgt_set
        self.edge_set : set[tuple[str, str]] = edge_set

    def subset(self, sub_tf_set, sub_tg_set):
        if sub_tf_set and sub_tg_set is None:
            stf_set = set([tfx for tfx in sub_tf_set if tfx in self.tf_set])
            sedge_set = set(
                [(tfx, tgx) for tfx, tgx in self.edge_set if tfx in stf_set]
            )
            stgt_set = set([tgx for _, tgx in sedge_set])
            return TFTGTNet(stf_set, stgt_set, sedge_set)
        if sub_tg_set and sub_tf_set is None:
            stgt_set = set([tfx for tfx in sub_tg_set if tfx in self.tgt_set])
            sedge_set = set(
                [(tfx, tgx) for tfx, tgx in self.edge_set if tgx in stgt_set]
            )
            stf_set = set([tfx for tfx, _ in sedge_set])
            return TFTGTNet(stf_set, stgt_set, sedge_set)
        elif sub_tf_set and sub_tg_set:
            stf_set = set([tfx for tfx in sub_tf_set if tfx in self.tf_set])
            stg_set = set([tfx for tfx in sub_tg_set if tfx in self.tgt_set])
            sedge_set = set(
                [
                    (tfx, tgx)
                    for tfx, tgx in self.edge_set
                    if tfx in stf_set and tgx in stg_set
                ]
            )
            stgt_set = set([tgx for _, tgx in sedge_set])
            return TFTGTNet(stf_set, stgt_set, sedge_set)
        return TFTGTNet(self.tf_set, self.tgt_set, self.edge_set)

    def print(self, pr_title: str):
        print("{}:[TF: {} ; TGT: {}; EDGES: {}]".format(
              pr_title,
              len(self.tf_set),
              len(self.tgt_set),
              len(self.edge_set)
        ))

    def write(self, output_file_name: str):
        with open(output_file_name, "w") as wfx:
            for tf, tgt in self.edge_set:
                wfx.write(" ".join([tf, tgt]))
                wfx.write("\n")


def imp_filter(threshold):
    return (
        pl.col("importance").is_not_null() &
        pl.col("importance").is_not_nan() &
        (pl.col("importance") > threshold) 
    )


def load_network(network_file, threshold):
    return pl.read_csv(network_file).select(
        pl.all().name.to_lowercase()
    ).filter(
        imp_filter(threshold)
    ).select([
        pl.col("tf"),
        pl.col("target"),
        pl.col("importance")
    ])


def get_network_edges(network_df: pl.DataFrame, ntop_edges: int):
    tf_set = set(network_df.get_column(network_df.columns[0]))
    tgt_set = set(network_df.get_column(network_df.columns[1]))
    edge_list = list(network_df.iter_rows())
    if ntop_edges is not None and ntop_edges > 0 and len(edge_list) > ntop_edges:
        edge_list.sort(key=lambda x: x[2], reverse=True)
        edge_list = edge_list[:ntop_edges]
    return tf_set, tgt_set, edge_list



def get_tftg_network(net_file: pathlib.Path, auc_args: AUCArgs):
    network_df = load_network(net_file, auc_args.threshold)
    tf_set, tgt_set, reg_edge_list  = get_network_edges(
        network_df, auc_args.ntop_edges
    )
    reg_edge_set = set((x, y) for x, y, _ in reg_edge_list)
    reg_net = TFTGTNet(tf_set, tgt_set, reg_edge_set)
    if len(auc_args.restrict_tf_set) > 0 or len(auc_args.restrict_tgt_set) > 0:
        print("RESTRICT", len(auc_args.restrict_tf_set), len(auc_args.restrict_tgt_set))
        reg_net = reg_net.subset(auc_args.restrict_tf_set, auc_args.restrict_tgt_set)
    return reg_net


def get_ground_truth_network(
    ground_net_file: pathlib.Path, ground_net_colums: list[str]
) -> TFTGTNet:
    if ground_net_file.suffix == ".csv":
        gt_net_df = pl.read_csv(ground_net_file)
    else:
        gt_net_df = pl.read_excel(ground_net_file, columns=ground_net_colums)
    edge_set = set(gt_net_df.select('TF','TARGET').iter_rows())
    tf_set = set(gt_net_df.get_column('TF'))
    tgt_set = set(gt_net_df.get_column('TARGET'))
    return TFTGTNet(tf_set, tgt_set, edge_set)


def get_trrust_network(trrust_file: pathlib.Path) -> TFTGTNet:
    return get_ground_truth_network(trrust_file, TRRUST_COLUMNS)


def get_pbmc_blood_network(pbmc_blood_file: pathlib.Path) -> TFTGTNet:
    return get_ground_truth_network(pbmc_blood_file, PBMC_BLOOD_COLUMNS)


def get_pbmc_blood_trrust_network(pbmc_blood_trrust_file: pathlib.Path) -> TFTGTNet:
    return get_ground_truth_network(pbmc_blood_trrust_file, PBMC_BLOOD_TRRUST_COLUMNS)


def get_pbmc_network(pbmc_file: pathlib.Path) -> TFTGTNet:
    return get_ground_truth_network(pbmc_file, PBMC_COLUMNS)


def generate_complete_network(tf_tgt_net: TFTGTNet):
    edge_set = list(itertools.product(tf_tgt_net.tf_set, tf_tgt_net.tgt_set))
    return TFTGTNet(tf_tgt_net.tf_set, tf_tgt_net.tgt_set, edge_set)


def get_edge_indicator(src_net: TFTGTNet, tgt_net: TFTGTNet):
    return [1 if x in tgt_net.edge_set else 0 for x in src_net.edge_set]


def init_networks(
    net_file: pathlib.Path,
    groundt_net: TFTGTNet,
    complete_net: TFTGTNet,
    auc_args: AUCArgs,
):
    reg_net = get_tftg_network(
        net_file,
        auc_args,
    )
    if auc_args.subset_tfs and auc_args.subset_tgts:
        groundt_net = groundt_net.subset(reg_net.tf_set, reg_net.tgt_set)
        complete_net = generate_complete_network(groundt_net)
    elif auc_args.subset_tgts:
        groundt_net = groundt_net.subset(None, reg_net.tgt_set)
        complete_net = generate_complete_network(groundt_net)
    elif auc_args.subset_tfs:
        groundt_net = groundt_net.subset(reg_net.tf_set, None)
        complete_net = generate_complete_network(groundt_net)
    return reg_net, groundt_net, complete_net


def get_net_tp(
    groundt_net: TFTGTNet,
    complete_net: TFTGTNet,
    net_file: pathlib.Path,
    auc_args: AUCArgs,
):
    reg_net, groundt_net, complete_net = init_networks(
        net_file, groundt_net, complete_net, auc_args
    )
    return groundt_net.edge_set & reg_net.edge_set


def get_auc_roc_pr(
    groundt_net: TFTGTNet,
    complete_net: TFTGTNet,
    index_key: str,
    net_file: pathlib.Path,
    auc_args: AUCArgs,
):
    reg_net, groundt_net, complete_net = init_networks(
        net_file, groundt_net, complete_net, auc_args
    )
    trrust_ind = get_edge_indicator(complete_net, groundt_net)
    reg_ind = get_edge_indicator(complete_net, reg_net)
    f1_score = sklearn.metrics.f1_score(trrust_ind, reg_ind)
    pr_score = sklearn.metrics.precision_score(trrust_ind, reg_ind)
    rc_score = sklearn.metrics.recall_score(trrust_ind, reg_ind)
    pbmc_blood_net = sklearn.metrics.roc_auc_score(trrust_ind, reg_ind)
    prc = sklearn.metrics.average_precision_score(trrust_ind, reg_ind)
    ncommon = len(groundt_net.edge_set & reg_net.edge_set)
    nedges = len(reg_net.edge_set)
    ntotaltp = len(groundt_net.edge_set)
    ratio = np.nan if nedges == 0 else float(ncommon) / float(nedges)
    return {
        "NETWORK": index_key,
        "EDGES": nedges,
        #  "C": len(trrust_ind),
        #  "D": sum(trrust_ind),
        # "EDGE_IND": sum(reg_ind),
        "TRUE_TP": ntotaltp,
        "NET_TP": ncommon,
        "RATIO": ratio,
        "F1": f1_score,
        "PREC": pr_score,
        "RECALL": rc_score,
        "AUROC": pbmc_blood_net,
        "AUPR": prc,
    }


def print_df(ndf: pl.DataFrame):
    with pl.Config() as cfg:
        cfg.set_tbl_cols(-1)
        cfg.set_tbl_rows(-1)
        cfg.set_tbl_width_chars(-1)
        cfg.set_float_precision(8)
        print(ndf)
    # # a
    # with pd.option_context(
    #     "display.max_rows",
    #     None,
    #     "display.max_columns",
    #     None,
    #     "display.width",
    #     None,
    #     "display.precision",
    #     8,
    # ):
    #     print(ndf)

def get_file_names(
    data_files_map, entry: str, index: int
) -> tuple[pathlib.Path, pathlib.Path]:
    file_dir = data_files_map[entry][index]["dir"]
    stats_file = pathlib.Path(file_dir, data_files_map[entry][index]["stats"])
    adj_file = pathlib.Path(file_dir, data_files_map[entry][index]["adj"])
    # print(loom_file, reg_file, adj_file)
    return stats_file, adj_file


def tftgt_net_stats(stats_file):
    with open(stats_file) as ifx:
        return json.load(ifx)


def roc_pr_analyses(
    network_files_map: dict[str, list[dict[str, str]]],
    groundt_net: TFTGTNet,
    complete_net: TFTGTNet,
    auc_args: AUCArgs,
):
    df_list = []
    for method_key in network_files_map.keys():
        for index in range(len(network_files_map[method_key])):
            stats_file, reg_file = get_file_names(
                network_files_map, method_key, index
            )
            run_stats = tftgt_net_stats(stats_file)
            net_stats = get_auc_roc_pr(
                groundt_net, complete_net, method_key, reg_file, auc_args
            )
            if "total_run_time" in run_stats:
                run_time = run_stats["total_run_time"]
                net_stats["RUN_TIME"] = run_time
                net_stats["RUN_TIME_HRS"] = run_time / 3600.0
            df_list.append(net_stats)
    net_df = pl.DataFrame(df_list)
    return net_df


def get_net_tp_union_network(
    netfile_data_map: dict[str, list[dict[str, str]]],
    groundt_net: TFTGTNet,
    complete_net: TFTGTNet,
    auc_args: AUCArgs,
):
    union_edge_set = set([])
    for method_key in netfile_data_map.keys():
        for index in range(len(netfile_data_map[method_key])):
            _, net_file = get_file_names(netfile_data_map, method_key, index)
            net_tp_edges = get_net_tp(groundt_net, complete_net, net_file, auc_args)
            union_edge_set = union_edge_set | net_tp_edges
    tf_set = set([tfx for tfx, _ in union_edge_set])
    tg_set = set([tgx for _, tgx in union_edge_set])
    return TFTGTNet(tf_set, tg_set, union_edge_set)


def get_intersect_tfs_targets(
    network_files_map: dict[str, list[dict[str, str]]],
):
    tf_set = set([])
    tgt_set = set([])
    for method_key in network_files_map.keys():
        for index in range(len(network_files_map[method_key])):
            _, net_file = get_file_names(network_files_map, method_key, index)
            reg_net = get_tftg_network(net_file, AUCArgs())
            if len(tf_set) == 0:
                tf_set = tf_set | reg_net.tf_set
                tgt_set = tgt_set | reg_net.tgt_set
            else:
                tf_set = tf_set & reg_net.tf_set
                tgt_set = tgt_set & reg_net.tgt_set
    return tf_set, tgt_set


def get_union_tfs_targets(
    network_files_map: dict[str, list[dict[str, str]]],
):
    tf_set = set([])
    tgt_set = set([])
    for method_key in network_files_map.keys():
        for index in range(len(network_files_map[method_key])):
            _, net_file = get_file_names(network_files_map, method_key, index)
            reg_net = get_tftg_network(net_file, AUCArgs())
            tf_set = tf_set | reg_net.tf_set
            tgt_set = tgt_set | reg_net.tgt_set
    return tf_set, tgt_set



def pbmc_tp_union_network(
    network_file_map,
    pbmc_net,
    complete_net,
):
    ntp_union_net = get_net_tp_union_network(
        network_file_map,
        pbmc_net,
        complete_net,
        AUCArgs(),
    )
    ntp_union_net.print("NTP UNION")
    ntp_union_complete_net = generate_complete_network(ntp_union_net)
    ntp_union_complete_net.print("NTP UNION COMPLETE")
    return ntp_union_net, ntp_union_complete_net


def pbmc_true_network(network_file_map):
    int_tf_set, int_tgt_set = get_intersect_tfs_targets(network_file_map)
    union_tf_set, union_tgt_set = get_union_tfs_targets(network_file_map)
    print("TRUE ISECT:[ TF : {}; TGT: {}] ; UNION:[ TF : {}; TGT: {}]".format(
        len(int_tf_set),
        len(int_tgt_set),
        len(union_tf_set),
        len(union_tgt_set),
    ))
    pbmc_net = get_pbmc_network(PBMC_NET_FILE)
    pbmc_net.print("PBMC")
    pbmc_net = pbmc_net.subset(union_tf_set, union_tgt_set)
    pbmc_net.print("PBMC")
    complete_net = generate_complete_network(pbmc_net)
    complete_net.print("PBMC COMPLETE")
    return pbmc_net, complete_net


def analyses_wrt_pbmc(network_file_map):
    pbmc_net, complete_net = pbmc_true_network(network_file_map)
    ntp_union_net, ntp_union_complete_net = pbmc_blood_tp_union_network(
        network_file_map, pbmc_net, complete_net
    )
    net_df1 = roc_pr_analyses(
        network_file_map,
        pbmc_net,
        complete_net,
        AUCArgs(),
    )
    net_df2 = roc_pr_analyses(
        network_file_map,
        ntp_union_net,
        ntp_union_complete_net,
        AUCArgs(),
    )
    print("""----- PBMC ------""")
    print("""----- FULL ------""")
    print_df(net_df1)
    print("""----- NTP UNION FULL ------""")
    print_df(net_df2)


def pbmc_blood_true_network(network_file_map):
    int_tf_set, int_tgt_set = get_intersect_tfs_targets(network_file_map)
    union_tf_set, union_tgt_set = get_union_tfs_targets(network_file_map)
    print("TRUE ISECT:[ TF : {}; TGT: {}] ; UNION:[ TF : {}; TGT: {}]".format(
        len(int_tf_set),
        len(int_tgt_set),
        len(union_tf_set),
        len(union_tgt_set),
    ))
    pbmc_blood_net = get_pbmc_blood_network(PBMC_BLOOD_NET_FILE)
    pbmc_blood_net.print("PBMC BLOOD")
    pbmc_blood_net = pbmc_blood_net.subset(union_tf_set, union_tgt_set)
    pbmc_blood_net.print("PBMC BLOOD")
    complete_net = generate_complete_network(pbmc_blood_net)
    complete_net.print("PBMC BLOOD COMPLETE")
    return pbmc_blood_net, complete_net


def pbmc_blood_tp_union_network(
    network_file_map,
    pbmc_blood_net,
    complete_net,
):
    ntp_union_net = get_net_tp_union_network(
        network_file_map,
        pbmc_blood_net,
        complete_net,
        AUCArgs(),
    )
    ntp_union_net.print("NTP UNION")
    ntp_union_complete_net = generate_complete_network(ntp_union_net)
    ntp_union_complete_net.print("NTP UNION COMPLETE")
    return ntp_union_net, ntp_union_complete_net

def analyses_wrt_pbmc_blood(network_file_map):
    pbmc_blood_net, complete_net = pbmc_blood_true_network(network_file_map)
    ntp_union_net, ntp_union_complete_net = pbmc_blood_tp_union_network(
        network_file_map, pbmc_blood_net, complete_net
    )
    net_df1 = roc_pr_analyses(
        network_file_map,
        pbmc_blood_net,
        complete_net,
        AUCArgs(),
    )
    net_df2 = roc_pr_analyses(
        network_file_map,
        ntp_union_net,
        ntp_union_complete_net,
        AUCArgs(),
    )
    print("""----- PBMC BLOOD ------""")
    print("""----- FULL ------""")
    print_df(net_df1)
    print("""----- NTP UNION FULL ------""")
    print_df(net_df2)

def pbmc_trrust_true_network(network_file_map):
    int_tf_set, int_tgt_set = get_intersect_tfs_targets(network_file_map)
    union_tf_set, union_tgt_set = get_union_tfs_targets(network_file_map)
    print("TRUE ISECT:[ TF : {}; TGT: {}] ; UNION:[ TF : {}; TGT: {}]".format(
        len(int_tf_set),
        len(int_tgt_set),
        len(union_tf_set),
        len(union_tgt_set),
    ))
    pbmc_trrust_net = get_pbmc_blood_network(PBMC_TRRUST_NET_FILE)
    pbmc_trrust_net.print("PBMC TRRUST")
    pbmc_trrust_net = pbmc_trrust_net.subset(union_tf_set, union_tgt_set)
    pbmc_trrust_net.print("PBMC TRRUST")
    complete_net = generate_complete_network(pbmc_trrust_net)
    complete_net.print("PBMC TRRUST COMPLETE")
    return pbmc_trrust_net, complete_net


def pbmc_trrust_tp_union_network(
    network_file_map,
    pbmc_trrust_net,
    complete_net,
):
    ntp_union_net = get_net_tp_union_network(
        network_file_map,
        pbmc_trrust_net,
        complete_net,
        AUCArgs(),
    )
    ntp_union_net.print("NTP UNION")
    ntp_union_complete_net = generate_complete_network(ntp_union_net)
    ntp_union_complete_net.print("NTP UNION COMPLETE")
    return ntp_union_net, ntp_union_complete_net


def analyses_wrt_pbmc_trrust(
    network_file_map,
    ntop_edges=100000
):
    pbmc_trrust_net, complete_net = pbmc_trrust_true_network(network_file_map)
    ntp_union_net, ntp_union_complete_net = pbmc_trrust_tp_union_network(
        network_file_map, pbmc_trrust_net, complete_net
    )
    #
    net_df1 = roc_pr_analyses(
        network_file_map,
        pbmc_trrust_net,
        complete_net,
        AUCArgs(),
    )
    net_df2 = roc_pr_analyses(
        network_file_map,
        ntp_union_net,
        ntp_union_complete_net,
        AUCArgs(),
    )
    net_df3 = roc_pr_analyses(
        network_file_map,
        ntp_union_net,
        ntp_union_complete_net,
        AUCArgs(ntop_edges=ntop_edges),
    )
    print("""----- PBMC TRRUST ------""")
    print("""----- FULL ------""")
    print_df(net_df1)
    print("""----- NTP FULL ------""")
    print_df(net_df2)
    print(f"""----- NTP UNION {ntop_edges}  ------""")
    print_df(net_df3)


def pbmc_blood_trrust_true_network(network_file_map):
    int_tf_set, int_tgt_set = get_intersect_tfs_targets(network_file_map)
    print("INTERSECT TF SET : ", len(int_tf_set))
    print("INTERSECT TGT SET : ", len(int_tgt_set))
    union_tf_set, union_tgt_set = get_union_tfs_targets(network_file_map)
    print("UNION TF SET : ", len(union_tf_set))
    print("UNION TGT SET : ", len(union_tgt_set))
    pbmc_blood_trrust_net = get_pbmc_blood_trrust_network(PBMC_BLOOD_TRRUST_NET_FILE)
    pbmc_blood_trrust_net.print("PBMC BLOOD")
    pbmc_blood_trrust_net = pbmc_blood_trrust_net.subset(union_tf_set, union_tgt_set)
    pbmc_blood_trrust_net.print("PBMC BLOOD")
    complete_net = generate_complete_network(pbmc_blood_trrust_net)
    complete_net.print("PBMC BLOOD COMPLETE")
    return pbmc_blood_trrust_net, complete_net


def pbmc_blood_trrust_tp_union_network(
    network_file_map,
    pbmc_blood_trrust_net,
    complete_net,
):
    ntp_union_net = get_net_tp_union_network(
        network_file_map,
        pbmc_blood_trrust_net,
        complete_net,
        AUCArgs(),
    )
    ntp_union_net.print("NTP UNION")
    ntp_union_complete_net = generate_complete_network(ntp_union_net)
    ntp_union_complete_net.print("NTP UNION COMPLETE")
    return ntp_union_net, ntp_union_complete_net


def analyses_wrt_pbmc_blood_trrust(
    network_file_map,
    ntop_edges=10000
):
    pbmc_blood_trrust_net, complete_net = pbmc_blood_trrust_true_network(
        network_file_map
    )
    ntp_union_net, ntp_union_complete_net = pbmc_blood_trrust_tp_union_network(
        network_file_map, pbmc_blood_trrust_net, complete_net
    )
    net_df1 = roc_pr_analyses(
        network_file_map,
        pbmc_blood_trrust_net,
        complete_net,
        AUCArgs(),
    )
    net_df2 = roc_pr_analyses(
        network_file_map,
        ntp_union_net,
        ntp_union_complete_net,
        AUCArgs(),
    )
    net_df3 = roc_pr_analyses(
        network_file_map,
        ntp_union_net,
        ntp_union_complete_net,
        AUCArgs(ntop_edges=ntop_edges),
    )
    print("""----- PBMC BLOOD TRRUST ------""")
    print("""----- FULL ------""")
    print_df(net_df1)
    print("""----- NTP UNION FULL ------""")
    print_df(net_df2)
    print(f"""----- NTP UNION {ntop_edges}  ------""")
    print_df(net_df3)

def net_ifull_analyses():
    gset_file_out_map = {
        "pbmc": "./genes_list/pbmc_genes.txt",
        "pbmc_ntp": "./genes_list/pbmc_ntp_genes.txt",
        "pbmc_blood": "./genes_list/pbmc_blood_genes.txt",
        "pbmc_blood_ntp": "./genes_list/pbmc_blood_ntp_genes.txt",
        "pbmc_trrust": "./genes_list/pbmc_trrust_genes.txt",
        "pbmc_trrust_ntp": "./genes_list/pbmc_trrust_ntp_genes.txt",
        "pbmc_blood_trrust": "./genes_list/pbmc_blood_trrust_genes.txt",
        "pbmc_blood_trrust_ntp": "./genes_list/pbmc_blood_trrust_ntp_genes.txt",
    }
    # generate_gene_sets(PYSCENIC_FULL_OUT_DATA, gset_file_out_map)
    # trrust_subset_network()
    # analyses_wrt_pbmc(NETWORK_DATA_FILES)
    # analyses_wrt_pbmc_blood(NETWORK_DATA_FILES)
    # analyses_wrt_pbmc_trrust(NETWORK_DATA_FILES)
    # analyses_wrt_pbmc_blood_trrust(NETWORK_DATA_FILES)
    analyses_wrt_pbmc_trrust(NETWORK20K_DATA_FILES)
    analyses_wrt_pbmc_blood_trrust(NETWORK20K_DATA_FILES, ntop_edges=100000)

def main():
    net_ifull_analyses()


if __name__ == "__main__":
    main()
