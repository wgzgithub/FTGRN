import numpy as np
import pandas as pd

def process_grn_data(adata):
    """
    将 adata.uns["grn"] 中的 Gene1 和 Gene2 转换为边索引和基因映射字典。

    参数:
    adata: AnnData 对象，其中 uns["grn"] 包含 Gene1 和 Gene2 列的 DataFrame

    返回:
    tuple: (gene_pairs, gene_edge_index, gene_to_idx)
        - gene_pairs: List[Tuple], 形如 [(gene1, gene2), ...] 的基因对列表
        - gene_edge_index: numpy.ndarray, 形状为 (2, N) 的边索引矩阵
        - gene_to_idx: Dict, 基因到索引的映射字典
    """
    grn_df = adata.uns["grn"]
    unique_genes = sorted(set(grn_df["Gene1"].tolist() + grn_df["Gene2"].tolist()))
    gene_to_idx = {gene: idx for idx, gene in enumerate(unique_genes)}
    gene_pairs = list(zip(grn_df["Gene1"], grn_df["Gene2"]))
    gene_edge_index = np.array([
        [gene_to_idx[gene1] for gene1 in grn_df["Gene1"]],
        [gene_to_idx[gene2] for gene2 in grn_df["Gene2"]]
    ])
    return gene_pairs, gene_edge_index, gene_to_idx

def compare_grns(grn, adata_grn):
    """
    比较两个基因调控网络 (GRN)，计算它们的交集和差集，以及相关统计信息。

    参数:
    grn: pandas.DataFrame, 参考 GRN
    adata_grn: pandas.DataFrame, 待比较的 GRN

    返回:
    dict: 包含以下键值对的字典:
        - overlap_ratio: float, 交集占 adata_grn 的比例
        - overlap_pairs: DataFrame, 两个网络中共有的调控对
        - unique_pairs: DataFrame, 仅在 adata_grn 中出现的调控对
        - stats: dict, 包含基本统计信息
    """
    assert 'Gene1' in grn.columns and 'Gene2' in grn.columns, "参考 GRN 必须包含 Gene1 和 Gene2 列"
    assert 'Gene1' in adata_grn.columns and 'Gene2' in adata_grn.columns, "待比较 GRN 必须包含 Gene1 和 Gene2 列"
    
    grn['pair'] = grn['Gene1'] + '_' + grn['Gene2']
    adata_grn['pair'] = adata_grn['Gene1'] + '_' + adata_grn['Gene2']
    
    overlap_pairs_str = set(adata_grn['pair']).intersection(set(grn['pair']))
    overlap_ratio = len(overlap_pairs_str) / len(adata_grn) * 100
    overlap_pairs = adata_grn[adata_grn['pair'].isin(overlap_pairs_str)][['Gene1', 'Gene2']]
    unique_pairs = adata_grn[~adata_grn['pair'].isin(overlap_pairs_str)][['Gene1', 'Gene2']]
    
    stats = {
        'total_grn_pairs': len(grn),
        'total_adata_pairs': len(adata_grn),
        'overlap_pairs_count': len(overlap_pairs),
        'unique_pairs_count': len(unique_pairs)
    }
    
    return {
        'overlap_ratio': overlap_ratio,
        'overlap_pairs': overlap_pairs,
        'unique_pairs': unique_pairs,
        'stats': stats
    }