import abc
import typing as t
#
import numpy as np
import numpy.typing as npt
import scanpy as sc
import pandas as pd
#
from anndata import AnnData
from pydantic import BaseModel

NDFloatArray : t.TypeAlias = npt.NDArray[np.floating[t.Any]]

def idx_dict(slist: list[str]) -> dict[str, int]:
    return dict(zip(slist, range(len(slist))))


class SCDataArgs(BaseModel):
    format: t.Literal['ad'] = 'ad'
    data : t.Literal['sparse', 'dense'] = 'dense'
    tf_file : str = "./trrust_tf.txt"
    h5ad_file : str = "./../rsc/h5/adata.raw.h5ad"
    select_hvg : int = True
    ntop_genes : int = 2000
    scale_regress : bool=True
    nsub_cells : int  = 0


class CSVDataArgs(BaseModel):
    format: t.Literal['csv'] = 'csv'
    csv_file : str = ""
    nsub_cells : int  = 0


class ExpDataProcessor(abc.ABC):
    def __init__(self) -> None:
        self.ntfs_ : int = 0
        self.ngenes_ : int = 0
        self.gene_map_ : dict[str, int] = {}
        self.tf_map_ : dict[str, int] = {}
        self.gene_list_ : list[str] = []
        self.tf_list_ : list[str] = []

    @property
    @abc.abstractmethod
    def exp_matrix(self) -> NDFloatArray:
        pass

    @property
    @abc.abstractmethod
    def tf_exp_matrix(self) -> NDFloatArray:
        pass

    @property
    def ntfs(self) -> int:
        return self.ntfs_

    @property
    def ngenes(self) -> int:
        return self.ngenes_

    @property
    def gene_map(self) -> dict[str, int]:
        return self.gene_map_

    @property
    def tf_map(self) -> dict[str, int]:
        return self.tf_map_

    @property
    def gene_list(self) -> list[str]:
        return self.gene_list_

    @property
    def tf_list(self) -> list[str]:
        return self.tf_list_

    def print(self):
        print(f"""
            No. Genes             : {self.ngenes}
            No. TF                : {self.ntfs}
            Expt Matrix shape     : {self.exp_matrix.shape}
            TF Expt Matrix  shape : {self.tf_exp_matrix.shape}
        """)

class CSVDataProcessor(ExpDataProcessor):
    def __init__(self, sargs: CSVDataArgs) -> None:
        super().__init__()
        self.adata_ : pd.DataFrame = pd.read_csv(
            sargs.csv_file, header=0, index_col=0
        )
        self.ematrix_ : NDFloatArray = self.adata_.T.to_numpy()
        #
        self.gene_list_ : list[str] = list(self.adata_.index)
        self.gene_map_ : dict[str, int]= idx_dict(self.gene_list_)
        self.ngenes_ : int = len(self.gene_list_)
        #
        self.tf_list_ : list[str] = list(self.adata_.index)
        self.tf_map_ : dict[str, int]= idx_dict(self.tf_list_)
        self.ntfs_ : int = len(self.tf_list_)

    @property
    @t.override
    def exp_matrix(self) -> NDFloatArray:
        return self.ematrix_

    @property
    @t.override
    def tf_exp_matrix(self) -> NDFloatArray:
        return self.ematrix_


class SCDataProcessor(ExpDataProcessor):
    def __init__(self, sargs: SCDataArgs) -> None:
        super().__init__()
        pdf = pd.read_csv(sargs.tf_file)
        self.all_tf_list_ : list[str] = list(pdf.gene)
        self.adata_ : AnnData = sc.read_h5ad(sargs.h5ad_file)
        # QC Metrics
        self.adata_.var["mt"] = self.adata_.var_names.str.startswith("MT-")
        self.adata_.var["rb"] = self.adata_.var_names.str.startswith("RPS")
        sc.pp.calculate_qc_metrics(
            self.adata_,
            qc_vars=["mt", "rb"],
            percent_top=None,
            log1p=False,
            inplace=True
        )
        #  Subset cells,
        ncells = self.adata_.shape[0]
        nsub_cells = sargs.nsub_cells
        if nsub_cells > 0 and nsub_cells < ncells:
            np.random.seed(0)
            self.adata_ = self.adata_[np.random.choice(ncells, nsub_cells), :]
        # Normalization
        sc.pp.normalize_total(self.adata_, target_sum=int(1e4))
        sc.pp.log1p(self.adata_)
        # Highly variable genes
        sc.pp.highly_variable_genes(self.adata_, n_top_genes=sargs.ntop_genes)
        if sargs.select_hvg:
            self.adata_.raw = self.adata_
            # Restrict to highly variable genes
            self.adata_ = self.adata_[:, self.adata_.var.highly_variable]
            if sargs.scale_regress:
                # Regress and Scale data
                sc.pp.regress_out(self.adata_,
                                  keys=["total_counts", "pct_counts_mt"])
                sc.pp.scale(self.adata_, max_value=10)
        else:
            print("Not selecting highly_variable_genes")
        #
        # Gene list
        self.gene_list_ : list[str] = list(self.adata_.var.index)
        self.gene_map_ : dict[str, int]= idx_dict(self.gene_list_)
        self.ngenes_ : int = len(self.gene_list_)
        #
        # TF list
        all_tf_set = set(self.all_tf_list_)
        self.tf_list_ : list[str] = list(
            tx for tx in all_tf_set if tx in self.gene_map_
        )
        self.ntfs_ : int = len(self.tf_list_)
        self.tf_map_ : dict[str, int]  = idx_dict(self.tf_list_)
        self.tf_indices_ : list[int] = list(
            self.gene_map_[tfx] for tfx in self.tf_list_
        )
        #
        # TF Anndata
        self.tf_adata_ : AnnData = (
            self.adata_[:, self.tf_list_] # pyright:ignore[reportArgumentType] 
        )

    @property
    @t.override
    def exp_matrix(self) -> NDFloatArray:
        return self.adata_.X  # pyright:ignore[reportReturnType]

    @property
    @t.override
    def tf_exp_matrix(self) -> NDFloatArray:
        return self.tf_adata_.X  # pyright: ignore[reportReturnType]

