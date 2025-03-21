import dataclasses as dcl
import itertools
import json
import typing as t
#
import numpy as np
import pandas as pd
import scipy.sparse
import xgboost as xgb
import lightgbm as lgb
import tqdm
import sklearn.ensemble as skm
import arboreto.algo
import arboreto.core
#
from timeit import default_timer as timer
#
from .data import ExpDataProcessor, NDFloatArray


DataArray : t.TypeAlias = (
    NDFloatArray |
    scipy.sparse.csr_matrix |
    scipy.sparse.csc_matrix
)


Regressor : t.TypeAlias = (
    skm.GradientBoostingRegressor |
    xgb.XGBModel                  |
    lgb.LGBMRegressor
)


RegressorMethod = t.Literal['xgb', 'sgbr', 'arb:default', 'arb:gbm', 'lgb'] 


def matrix_sub_row(
    in_matrix: DataArray,
    row : int
) -> DataArray:
    if isinstance(in_matrix, (scipy.sparse.csr_matrix,
                              scipy.sparse.csc_matrix)):
        return scipy.sparse.hstack( # pyright: ignore[reportReturnType]
            (
                in_matrix[:, :row], 
                in_matrix[:, row + 1 :]
            )
        )
    elif isinstance(
        in_matrix,
        np.ndarray
    ): # pyright: ignore[reportUnnecessaryIsInstance]
        return np.delete(in_matrix, row, 1)
    else:
        return in_matrix # pyright: ignore[reportUnreachable]


def matrix_column(
    in_matrix: DataArray,
    col : int
) -> DataArray:
    if isinstance(in_matrix, (scipy.sparse.csr_matrix,
                              scipy.sparse.csc_matrix)):
        return in_matrix[:, col].toarray().flatten()
    elif isinstance(
        in_matrix,
        np.ndarray
    ): # pyright: ignore[reportUnnecessaryIsInstance]
        return in_matrix[:, col].flatten()


@dcl.dataclass
class GMStats:
    run_time: float = 0.0
    n_features: float = 0.0


@t.final
class GBRunner:
    def __init__(self, expr_data: ExpDataProcessor) -> None:
        self.sd_ = expr_data
        # Output data
        self.mstats_ = [GMStats() for _ in range(self.sd_.ngenes)]
        self.importance_ : NDFloatArray | None = None
        self.run_time_ = 0.0
        self.run_desc_ = "GB Runner"

    def idx_dict(self, slist: list[str]) -> dict[str, int]:
        return dict(zip(slist, range(len(slist))))

    def target_gene_matrix(self, target_gene: str):
        if target_gene not in self.sd_.gene_map :
            return None, None
        if target_gene not in self.sd_.tf_map:
            sidx = self.sd_.gene_map[target_gene]
            return self.sd_.tf_exp_matrix, matrix_column(self.sd_.exp_matrix, sidx)
        tidx = self.sd_.tf_map[target_gene]
        return (matrix_sub_row(self.sd_.tf_exp_matrix, tidx),
                matrix_column(self.sd_.exp_matrix, tidx)) 

    def sgbr_fit(
        self,
        target_gene: str,
        **run_args: dict[str, t.Any]
    ) -> skm.GradientBoostingRegressor | None:
        exp_mat, tg_exp = self.target_gene_matrix(target_gene)
        if tg_exp is None:
            return None
        skl = skm.GradientBoostingRegressor(
            **run_args  # pyright: ignore[reportArgumentType]
        )
        return skl.fit(exp_mat, tg_exp)

    def xgb_fit(
        self,
        target_gene: str,
        **run_args: t.Any
    ) -> xgb.XGBModel | None:
        exp_mat, tg_exp = self.target_gene_matrix(target_gene)
        if tg_exp is None:
            return None
        xsr = xgb.XGBRegressor(**run_args)
        return xsr.fit(
            exp_mat,
            tg_exp
        ) 

    def lgb_fit(
        self,
        target_gene: str,
        **run_args: t.Any
    ) -> lgb.LGBMRegressor | None:
        exp_mat, tg_exp = self.target_gene_matrix(target_gene)
        if tg_exp is None:
            return None
        lsr = lgb.LGBMRegressor(**run_args)
        return lsr.fit(exp_mat, tg_exp)


    def fit(
        self,
        regressor_method: RegressorMethod,
        target_gene: str,
        **run_args: t.Any
    ) -> Regressor | None:
        match regressor_method:
            case 'xgb':
                return self.xgb_fit(target_gene, **run_args)
            case 'sgbr':
                return self.sgbr_fit(target_gene, **run_args)
            case 'lgb':
                return self.lgb_fit(target_gene, **run_args)
            case _:
                return None

    def update(self, gene: str, gmodel: Regressor, rtime: float):
        gidx = self.sd_.gene_map[gene]
        mfeat = gmodel.feature_importances_
        self.mstats_[gidx].run_time = rtime
        self.mstats_[gidx].n_features = int(np.sum(mfeat != 0.0))
        if gene in self.sd_.tf_map:
            mfeat = np.reshape(mfeat, shape=(1, self.sd_.ntfs-1))
            tidx = self.sd_.tf_map[gene]
            if tidx > 0:
                self.importance_[gidx, :tidx] = mfeat[:, :tidx]
            if tidx+1 < self.sd_.ntfs:
                self.importance_[gidx, tidx+1:] = mfeat[:, tidx:]
        else:
            self.importance_[gidx, :] = np.reshape(
                mfeat,
                shape=(1, self.sd_.ntfs)
            )

    def gene_model(
        self,
        method : RegressorMethod,
        tgene: str,
        **gb_args: t.Any
    ):
        start_time = timer() 
        mdx  = self.fit(method, tgene, **gb_args)
        if mdx:
            self.update(tgene, mdx, timer() - start_time)

    def init_importance(self, take_n: int | None=None):
        self.importance_  = np.zeros(
            shape=(take_n if take_n else self.sd_.ngenes, self.sd_.ntfs),
            dtype=np.float32
        )

    def genes_itr(self, take_n: int | None, use_tqdm: bool):
        giter = itertools.islice(self.sd_.gene_map.keys(),
                                 take_n if take_n else None)
        if use_tqdm:
            ttotal = take_n if take_n else len(self.sd_.gene_map)
            miniters = ttotal * 0.25
            return tqdm.tqdm(
                giter,
                desc=self.run_desc_,
                miniters=miniters,
                total=ttotal
            )
        else:
            return giter

    def build(
        self,
        method: RegressorMethod,
        take_n: int | None=None,
        use_tqdm: bool=True,
        **run_args: t.Any,
    ):
        start_time = timer() 
        self.init_importance(take_n)
        for target_gene in self.genes_itr(take_n, use_tqdm):
            self.gene_model(method, target_gene, **run_args)
        self.run_time_ = timer() - start_time
        print(f"Model Generation : {self.run_time_} seconds")
    
    def dump(
        self,
        rstats_out_file: str | None=None,
        out_file: str | None=None
    ):
        if rstats_out_file:
            with open(rstats_out_file, 'w') as ofx:
                json.dump({
                    "total_run_time": self.run_time_,
                    "model_data": [dcl.asdict(x) for x in self.mstats_]
                }, ofx, indent=4)
        if (out_file is not None ) and (self.importance_ is not None):
            im_shape = self.importance_.shape
            tf_tgt_itr = itertools.product(
                self.sd_.gene_list[:im_shape[0]],
                self.sd_.tf_list
            )
            rdf = pd.DataFrame(
                tf_tgt_itr,
                columns=pd.Series(["Target", "TF"])
            )
            rdf["importance"] = self.importance_.flatten()
            rdf.to_csv(out_file)


class ARBRunner:
    def __init__(self, exp_data: ExpDataProcessor) -> None:
        self.sd_ : ExpDataProcessor = exp_data
        self.run_time_ : float = 0.0
        self.network_ : pd.DataFrame | None = None

    def run_regression(
        self,
        method: RegressorMethod = 'arb:default',
        **run_args: t.Any,
    ) -> pd.DataFrame:
        match method:
            case 'arb:gbm':
                print(f"Running GBM with {arboreto.core.GBM_KWARGS}")
                return arboreto.algo.diy(
                    expression_data=self.sd_.exp_matrix,
                    regressor_type='GBM',
                    regressor_kwargs=arboreto.core.GBM_KWARGS,
                    gene_names=self.sd_.gene_list,
                    tf_names=self.sd_.tf_list, #pyright:ignore[reportArgumentType]
                )
            case "arb:default" | _ :
                print(f"Running GBM with default args")
                return arboreto.algo.grnboost2(
                    expression_data=self.sd_.exp_matrix,
                    gene_names=self.sd_.gene_list,
                    tf_names=self.sd_.tf_list, #pyright:ignore[reportArgumentType]
                )

    def build(
        self,
        method: RegressorMethod = 'arb:default',
        **run_args: t.Any
    ):
        start_time = timer()
        self.network_ = self.run_regression(method, **run_args)
        self.run_time_ = timer() - start_time
        print(f"Model Generation : {self.run_time_} seconds")

    def dump(
        self,
        rstats_out_file: str | None=None,
        out_file: str | None=None
    ):
        if rstats_out_file:
            with open(rstats_out_file, 'w') as ofx:
                json.dump({
                    "total_run_time": self.run_time_,
                }, ofx, indent=4)
        if (out_file is not None ) and (self.network_ is not None):
            self.network_.to_csv(out_file)


