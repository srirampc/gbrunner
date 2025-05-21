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
from pydantic import BaseModel
#
from .data import ExpDataProcessor, NDFloatArray


class XGBArgs(BaseModel):
    regressor: t.Literal['xgb'] = 'xgb' 
    learning_rate : float = 0.01
    n_estimators  : int = 500


class STXGBArgs(BaseModel):
    regressor: t.Literal['stxgb'] = 'stxgb' 
    learning_rate: float = 0.01
    n_estimators: int = 500  # can be arbitrarily large
    # max_features: float = 0.1
    colsample_bytree: float = 0.2 
    colsample_bynode: float = 0.5 
    subsample: float  = 0.9


class SGBMArgs(BaseModel): 
    regressor: t.Literal['sgbr'] = 'sgbr' 
    learning_rate : float = 0.01
    n_estimators  : float = 500
    max_features  : float = 0.1
    early_stop: bool = False


class STSGBMArgs(BaseModel): 
    regressor: t.Literal['stsgbr'] = 'stsgbr' 
    learning_rate : float = 0.01
    n_estimators  : float = 5000
    max_features  : float = 0.1
    subsample: float  = 0.9
    early_stop: bool = True


class LGBArgs(BaseModel):
    regressor: t.Literal['lgb'] = 'lgb' 
    n_estimators: int = 500
    learning_rate: float = 0.01
    n_jobs: int = 0
    verbosity: int = 0 
    force_col_wise: bool = False
    force_row_wise: bool = False
    objective: str = 'regression'
    importance_type: str = 'gain'


class STLGBArgs(BaseModel):
    regressor: t.Literal['stlgb'] = 'stlgb' 
    n_estimators: int = 500
    learning_rate: float = 0.01
    n_jobs: int = 0 
    verbosity: int = 0 
    force_col_wise: bool = False
    force_row_wise: bool = False
    colsample_bytree: float = 0.1 
    # colsample_bynode: float = 0.5 
    objective: str = 'regression'
    importance_type: str = 'gain'


GBRArgs : t.TypeAlias = (
    XGBArgs |
    STXGBArgs |
    SGBMArgs |
    STSGBMArgs |
    LGBArgs |
    STLGBArgs
)


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


GBMethod = t.Literal[
    'xgb', 'stxgb', 'sgbr', 'stsgbr', 'arb:default', 'arb:gbm', 'lgb', 'stlgb'
]

EARLY_STOP_WINDOW_LENGTH = 25

class EarlyStopMonitor:

    def __init__(self, window_length: int=EARLY_STOP_WINDOW_LENGTH):
        """
        :param window_length: length of the window over the out-of-bag errors.
        """

        self.window_length : int = window_length
        self.boost_rounds : int = 0

    def window_boundaries(self, current_round: int):
        """
        :param current_round:
        :return: the low and high boundaries of the estimators window to consider.
        """

        lo = max(0, current_round - self.window_length + 1)
        hi = current_round + 1

        return lo, hi

    def __call__(
        self,
        current_round: int,
        regressor: skm.GradientBoostingRegressor,
        _
    ):
        """
        Implementation of the GradientBoostingRegressor monitor function API.

        :param current_round: the current boosting round.
        :param regressor: the regressor.
        :param _: ignored.
        :return: True if the regressor should stop early, else False.
        """
        self.boost_rounds = current_round
        if current_round >= self.window_length - 1:
            lo, hi = self.window_boundaries(current_round)
            return np.mean(regressor.oob_improvement_[lo: hi]) < 0
        else:
            return False

def gbrunner_args(method: GBMethod) -> GBRArgs:
    match method:
        case 'sgbr':
            return SGBMArgs()
        case 'stsgbr':
            return STSGBMArgs()
        case 'stxgb':
            return STXGBArgs()
        case 'lgb':
            return LGBArgs()
        case 'stlgb':
            return STLGBArgs()
        case 'xgb':
            return XGBArgs()
        case 'arb:default' | 'arb:gbm':
            return XGBArgs()




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


class GMStats(BaseModel):
    run_time: float = 0.0
    fit_time: float = 0.0
    n_rounds: int = 0
    n_features: float = 0.0

class RunGMStats(BaseModel):
    total_run_time: float = 0.0
    gmodel_data: list[GMStats | None] = []

    @classmethod
    def init(cls, ngenes: int):
        return cls(gmodel_data=[None for _ in range(ngenes)])


@t.final
class GBRunner:
    def __init__(
        self,
        expr_data: ExpDataProcessor,
        device: t.Literal["cpu", "gpu"]
    ) -> None:
        self.sd_ = expr_data
        # Output data
        self.rstats_ = RunGMStats.init(self.sd_.ngenes)
        self.importance_ : NDFloatArray | None = None
        self.device_ = "cuda" if device == "gpu" else None
        # self.run_time_ = 0.0
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
        gb_args: SGBMArgs | STSGBMArgs,
    ) -> tuple[skm.GradientBoostingRegressor, GMStats] | None:
        exp_mat, tg_exp = self.target_gene_matrix(target_gene)
        if tg_exp is None:
            return None
        early_stop = EarlyStopMonitor() if gb_args.early_stop else None
        run_args = gb_args.model_dump(exclude=set(["regressor", "early_stop"]))
        start_time = timer() 
        skl = skm.GradientBoostingRegressor(**run_args).fit(
            exp_mat, tg_exp, monitor=early_stop
        )
        return skl, GMStats(
            fit_time=timer() - start_time,
            n_rounds=early_stop.boost_rounds if early_stop else 0
        )

    def xgb_fit(
        self,
        target_gene: str,
        gb_args: XGBArgs | STXGBArgs,
    ) -> tuple[xgb.XGBModel, GMStats] | None:
        exp_mat, tg_exp = self.target_gene_matrix(target_gene)
        if tg_exp is None:
            return None
        run_args = gb_args.model_dump(exclude=set(["regressor"]))
        if self.device_:
            run_args["device"] = "cuda"
        start_time = timer() 
        xsr = xgb.XGBRegressor(**run_args).fit(
            exp_mat,
            tg_exp
        )
        return xsr, GMStats(fit_time=timer() - start_time)
        

    def lgb_fit(
        self,
        target_gene: str,
        gb_args: LGBArgs | STLGBArgs,
    ) -> tuple[lgb.LGBMRegressor, GMStats] | None:
        exp_mat, tg_exp = self.target_gene_matrix(target_gene)
        if tg_exp is None:
            return None
        run_args = gb_args.model_dump(exclude=set(["regressor"]))
        start_time = timer() 
        lsr = lgb.LGBMRegressor(**run_args).fit(exp_mat, tg_exp)
        return lsr, GMStats(fit_time=timer() - start_time)


    def fit(
        self,
        regressor_method: GBMethod,
        target_gene: str,
        gb_args: GBRArgs 
    ) -> tuple[Regressor, GMStats] | None:
        match regressor_method:
            case 'xgb':
                return self.xgb_fit(target_gene, t.cast(XGBArgs, gb_args))
            case 'stxgb':
                return self.xgb_fit(target_gene, t.cast(STXGBArgs, gb_args))
            case 'sgbr':
                return self.sgbr_fit(target_gene, t.cast(SGBMArgs, gb_args))
            case 'stsgbr':
                return self.sgbr_fit(target_gene, t.cast(STSGBMArgs, gb_args))
            case 'lgb':
                return self.lgb_fit(target_gene, t.cast(LGBArgs,  gb_args))
            case 'stlgb':
                return self.lgb_fit(target_gene, t.cast(STLGBArgs,  gb_args))
            case _:
                return None

    def update(self, gene: str, gmodel: Regressor, gstat: GMStats):
        gidx = self.sd_.gene_map[gene]
        mfeat = gmodel.feature_importances_
        gstat.n_features = int(np.sum(mfeat != 0.0))
        self.rstats_.gmodel_data[gidx] = gstat
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
        method : GBMethod,
        tgene: str,
        gb_args: GBRArgs 
    ):
        start_time = timer() 
        fmdx  = self.fit(method, tgene, gb_args)
        if fmdx:
            mdx, gstat = fmdx
            gstat.run_time = timer() - start_time
            self.update(tgene, mdx, gstat)

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
        method: GBMethod,
        run_args: GBRArgs,
        take_n: int | None=None,
        use_tqdm: bool=True,
    ):
        start_time = timer() 
        self.init_importance(take_n)
        for target_gene in self.genes_itr(take_n, use_tqdm):
            self.gene_model(method, target_gene, run_args)
        self.rstats_.total_run_time = timer() - start_time
        print(f"Model Generation : {self.rstats_.total_run_time} seconds")
    
    def dump(
        self,
        rstats_out_file: str | None=None,
        out_file: str | None=None
    ):
        if rstats_out_file:
            with open(rstats_out_file, 'w') as ofx:
                ofx.write(self.rstats_.model_dump_json(indent=4))
        if (out_file is not None ) and (self.importance_ is not None):
            im_shape = self.importance_.shape
            tf_tgt_itr = itertools.product(
                self.sd_.gene_list[:im_shape[0]],
                self.sd_.tf_list
            )
            rdf = pd.DataFrame(
                tf_tgt_itr,
                columns=pd.Series(["target", "TF"])
            )
            rdf["importance"] = self.importance_.flatten()
            rdf.sort_values(by="importance", ascending=False, inplace=True)
            rdf.to_csv(out_file)


class ARBRunner:
    def __init__(self, exp_data: ExpDataProcessor) -> None:
        self.sd_ : ExpDataProcessor = exp_data
        self.run_time_ : float = 0.0
        self.network_ : pd.DataFrame | None = None

    def run_regression(
        self,
        method: GBMethod = 'arb:default',
        **_run_args: t.Any,
    ) -> pd.DataFrame:
        mat_shape = str(self.sd_.exp_matrix.shape)
        gene_l = str(len(self.sd_.gene_list)) 
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
                print(
                    f"Running GBM with default args {mat_shape} {gene_l}"
                )
                return arboreto.algo.grnboost2(
                    expression_data=self.sd_.exp_matrix,
                    gene_names=self.sd_.gene_list,
                    tf_names=self.sd_.tf_list, #pyright:ignore[reportArgumentType]
                )

    def build(
        self,
        method: GBMethod = 'arb:default',
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
