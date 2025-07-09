import torch
from models_pad import TransformerEncoder, LinkPredictor
from typing import Dict, Any

# 假设你的 TransformerEncoder 和 LinkPredictor 类定义在当前文件或已导入
# from your_model_file import TransformerEncoder, LinkPredictor # 确保导入正确的类定义
def save_model(
    encoder,
    predictor,
    optimizer,
    epoch,
    best_val_auc,  # 改为 best_val_auc 以匹配调用
    model_config,
    train_config,
    edge_index,
    train_pos_edge_index,
    gene_to_idx,
    save_path
):
    """
    保存模型和图数据结构

    Args:
        encoder: 编码器模型
        predictor: 预测器模型
        optimizer: 优化器
        epoch: 当前轮数
        best_val_auc: 最佳验证AUC
        model_config: 模型配置
        train_config: 训练配置
        edge_index: 图的边索引
        gene_to_idx: 基因到索引的映射字典
        save_path: 保存路径
    """
    # 使用 vars() 转换配置对象为字典
    model_config_dict = vars(model_config)
    train_config_dict = vars(train_config)

    torch.save({
        'encoder_state_dict': encoder.state_dict(),
        'predictor_state_dict': predictor.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch,
        'best_val_auc': best_val_auc,  # 改为 best_val_auc
        'model_config': model_config_dict,
        'train_config': train_config_dict,
        'edge_index': edge_index,
        'train_pos_edge_index': train_pos_edge_index,
        'gene_to_idx': gene_to_idx
    }, save_path)



def load_model(load_path, device=None):
    """加载模型"""
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"Loading model from {load_path} to {device}")
    checkpoint = torch.load(load_path, map_location=device)
    
    model_config_dict = checkpoint['model_config']
    
    # --- 关键修改：确定 original_feature_dim ---
    # 1. 尝试从模型配置中获取 (如果保存时添加了)
    original_feature_dim = model_config_dict.get('original_feature_dim', None)

    if original_feature_dim is None:
        # 2. 如果配置中没有，尝试从 learnable_fill_vector 的形状推断
        if 'learnable_fill_vector' in checkpoint['encoder_state_dict']:
            original_feature_dim = checkpoint['encoder_state_dict']['learnable_fill_vector'].size(1)
            print(f"Inferred original_feature_dim from learnable_fill_vector: {original_feature_dim}")
        else:
            # 3. 如果 learnable_fill_vector 也不在 (这不应该发生如果错误是 "Unexpected key")
            #    或者作为最后的手段，从第一个 conv 层的输入维度反推
            #    actual_in_channels = original_feature_dim + 1
            #    所以 original_feature_dim = actual_in_channels - 1
            conv_weight_key = [k for k in checkpoint['encoder_state_dict'].keys() 
                              if 'convs.0' in k and 'weight' in k][0] # 假设第一个卷积是 convs.0
            actual_in_channels_from_conv = checkpoint['encoder_state_dict'][conv_weight_key].size(1)
            original_feature_dim = actual_in_channels_from_conv - 1
            print(f"Inferred original_feature_dim by subtracting 1 from first conv input dim: {original_feature_dim} (from {actual_in_channels_from_conv})")

    if original_feature_dim is None:
        raise ValueError("Could not determine original_feature_dim for TransformerEncoder.")

    # 创建 TransformerEncoder 实例
    # 确保这里的 TransformerEncoder 是包含 learnable_fill_vector 的那个定义
    encoder = TransformerEncoder(
        original_feature_dim=original_feature_dim, # <--- 使用 original_feature_dim
        hidden_channels=model_config_dict['hidden_channels'],
        heads=model_config_dict['heads'],
        dropout=model_config_dict['dropout'],
        use_layer_norm=model_config_dict.get('use_layer_norm', True), # 使用 get 以兼容旧配置
        residual=model_config_dict.get('residual', True) # 使用 get 以兼容旧配置
    ).to(device)
    
    # 创建 LinkPredictor 实例 (这部分通常不变)
    predictor_input_dim = model_config_dict['hidden_channels'][-1]
    # 如果 encoder 的 hidden_channels 是空的（例如，一个非常简单的encoder），需要有备用逻辑
    if not model_config_dict['hidden_channels']:
        # 这种情况下，encoder的输出就是其输入（经过某些变换），或者需要一个默认值
        # 这取决于你的 encoder 设计，如果 hidden_channels 为空意味着什么
        # 一个可能的假设是 encoder 的输出维度等于 original_feature_dim (如果它只是个嵌入层或线性层)
        # 或者等于 learnable_fill_vector 的维度
        print(f"Warning: model_config_dict['hidden_channels'] is empty. Assuming predictor input dim is original_feature_dim ({original_feature_dim}).")
        predictor_input_dim = original_feature_dim # 或者更具体的逻辑


    predictor = LinkPredictor( # 确保 LinkPredictor 类也被正确导入
        in_channels=predictor_input_dim,
        hidden_channels=model_config_dict['pred_hidden_channels'],
        dropout=model_config_dict.get('pred_dropout', 0.1) # 使用 get 以兼容旧配置
    ).to(device)
    
    # 加载模型权重
    # 对于 encoder，由于 learnable_fill_vector 是在 __init__ 中定义的，
    # 只要类定义正确，state_dict 应该能正确加载。
    try:
        encoder.load_state_dict(checkpoint['encoder_state_dict'])
        print("Encoder state_dict loaded successfully.")
    except RuntimeError as e:
        print(f"RuntimeError loading encoder state_dict: {e}")
        print("Encoder state_dict keys in checkpoint:", checkpoint['encoder_state_dict'].keys())
        print("Encoder model parameters:", [name for name, _ in encoder.named_parameters()])
        raise e

    try:
        predictor.load_state_dict(checkpoint['predictor_state_dict'])
        print("Predictor state_dict loaded successfully.")
    except RuntimeError as e:
        print(f"RuntimeError loading predictor state_dict: {e}")
        raise e
        
    # ... (后续的 edge_index, gene_to_idx 等处理) ...
    if 'edge_index' in checkpoint and checkpoint['edge_index'] is not None:
        checkpoint['edge_index'] = checkpoint['edge_index'].to(device)

    if 'train_pos_edge_index' in checkpoint and checkpoint['train_pos_edge_index'] is not None:
        checkpoint['train_pos_edge_index'] = checkpoint['train_pos_edge_index'].to(device)
        
    # gene_to_idx 不是 tensor，不需要 to(device)
    # if 'gene_to_idx' in checkpoint:
    #     checkpoint['gene_to_idx'] = checkpoint['gene_to_idx'] # No change needed
        
    return encoder, predictor, checkpoint


import numpy as np
import pandas as pd

def filter_grn_by_genes_numpy(grn_df, adata):
    """
    使用 NumPy 过滤 GRN，只保留在 adata.var_names 中的基因对。
    """
    valid_genes = set(adata.var_names)
    genes1 = grn_df['Gene1'].values
    genes2 = grn_df['Gene2'].values
    mask1 = np.isin(genes1, list(valid_genes))
    mask2 = np.isin(genes2, list(valid_genes))
    final_mask = mask1 & mask2
    filtered_grn = grn_df[final_mask].copy()
    filtered_grn.reset_index(drop=True, inplace=True)
    return filtered_grn

