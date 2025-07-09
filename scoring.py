import numpy as np
from scipy import stats
import scipy
from sklearn.feature_selection import mutual_info_regression
from typing import List, Tuple, Dict, Optional
from enum import Enum
from dataclasses import dataclass

class ScoringMethod(Enum):
    MI = "mutual_information"
    SPEARMAN = "spearman"
    HYBRID = "hybrid"

@dataclass
class ScoringConfig:
    method: ScoringMethod
    n_top_genes: Optional[int] = None
    alpha: float = 0.01
    print_top: int = 10

class GRNScorer:
    def __init__(self, adata, gene_pairs: List[Tuple[str, str]], gene_list: Optional[List[str]] = None):
        self.adata = adata
        self.genes = adata.var_names
        self.gene_to_idx = {gene: i for i, gene in enumerate(self.genes)}
        
        if gene_list is not None:
            gene_set = set(gene_list)
            self.gene_pairs = [(tf, target) for tf, target in gene_pairs if tf in gene_set or target in gene_set]
            if not self.gene_pairs:
                raise ValueError("No valid gene pairs found after filtering with gene_list")
        else:
            invalid_genes = set()
            self.gene_pairs = []
            for tf, target in gene_pairs:
                if tf not in self.gene_to_idx:
                    invalid_genes.add(tf)
                if target not in self.gene_to_idx:
                    invalid_genes.add(target)
                if tf in self.gene_to_idx and target in self.gene_to_idx:
                    self.gene_pairs.append((tf, target))
            if invalid_genes:
                print(f"Warning: The following genes were not found: {', '.join(sorted(invalid_genes))}")
            if not self.gene_pairs:
                raise ValueError("No valid gene pairs found in the dataset")
        
        self.expr_matrix = self._get_expression_matrix()
        print(f"Initialized with {len(self.gene_pairs)} valid gene pairs")
    
    def _get_expression_matrix(self) -> np.ndarray:
        return self.adata.X.toarray() if scipy.sparse.issparse(self.adata.X) else self.adata.X
    
    def _calculate_mi_score(self, tf_expr: np.ndarray, target_expr: np.ndarray) -> float:
        tf_expr_2d = tf_expr.reshape(-1, 1)
        return mutual_info_regression(tf_expr_2d, target_expr, discrete_features=False)[0]
    
    def _calculate_spearman_score(self, tf_expr: np.ndarray, target_expr: np.ndarray) -> float:
        score, _ = stats.spearmanr(tf_expr, target_expr)
        return float('-inf') if np.isnan(score) else score
    
    def _calculate_hybrid_scores_batch(self, n_top_genes: int, alpha: float) -> Dict[Tuple[str, str], float]:
        ranks_matrix = np.apply_along_axis(lambda x: stats.rankdata(-x, method='dense'), 1, self.expr_matrix)
        scores = {}
        n_genes = len(self.genes)
        for tf, target in self.gene_pairs:
            tf_idx, target_idx = self.gene_to_idx[tf], self.gene_to_idx[target]
            tf_ranks, target_ranks = ranks_matrix[:, tf_idx], ranks_matrix[:, target_idx]
            tf_in_top = tf_ranks <= n_top_genes
            target_in_top = target_ranks <= n_top_genes
            both_in_top = tf_in_top & target_in_top
            tf_scores = np.where(both_in_top, 1 - (tf_ranks / n_top_genes), 0)
            target_scores = np.where(both_in_top, 1 - (target_ranks / n_top_genes), 0)
            base_scores = tf_scores * target_scores
            smooth_factors = np.exp(-alpha * (tf_ranks + target_ranks) / n_genes)
            final_scores = base_scores * smooth_factors
            scores[(tf, target)] = np.mean(final_scores)
        return scores
    
    def calculate_scores(self, config: ScoringConfig) -> Dict[Tuple[str, str], float]:
        scores = {}
        if config.method == ScoringMethod.HYBRID:
            n_top_genes = config.n_top_genes or int(len(self.genes) * 0.05)
            return self._calculate_hybrid_scores_batch(n_top_genes, config.alpha)
        for tf, target in self.gene_pairs:
            tf_idx, target_idx = self.gene_to_idx[tf], self.gene_to_idx[target]
            tf_expr, target_expr = self.expr_matrix[:, tf_idx], self.expr_matrix[:, target_idx]
            if config.method == ScoringMethod.MI:
                score = self._calculate_mi_score(tf_expr, target_expr)
            else:  # SPEARMAN
                score = self._calculate_spearman_score(tf_expr, target_expr)
            scores[(tf, target)] = score
        return scores
    
    def analyze(self, config: ScoringConfig) -> List[Tuple[Tuple[str, str], float]]:
        print(f"Calculating scores using {config.method.value} method...")
        scores = self.calculate_scores(config)
        sorted_pairs = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        print(f"\nTop {config.print_top} gene pairs:")
        for (tf, target), score in sorted_pairs[:config.print_top]:
            print(f"{tf} -> {target}: {'nan' if score == float('-inf') else f'{score:.4f}'}")
        nan_count = sum(1 for _, score in sorted_pairs if score == float('-inf'))
        if nan_count > 0:
            print(f"\nNote: Found {nan_count} pairs with invalid scores")
        return sorted_pairs