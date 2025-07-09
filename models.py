from torch_geometric.nn import TransformerConv
import torch
import torch.nn as nn
import torch.nn.functional as F

# Transformer编码器
class TransformerEncoder(nn.Module):
    def __init__(self, 
                 # in_channels, # 不再是固定的，而是 original_feature_dim + 1 (mask)
                 original_feature_dim, # 原始基因 embedding 的维度
                 hidden_channels,     # [1024, 512, 256]
                 heads=8,              
                 dropout=0.1,         
                 use_layer_norm=True,  
                 residual=True):       
        super().__init__()
        self.residual = residual
        self.use_layer_norm = use_layer_norm
        self.original_feature_dim = original_feature_dim

        # 可学习的填充向量，用于替换缺失的基因嵌入
        self.learnable_fill_vector = nn.Parameter(torch.Tensor(1, original_feature_dim))
        nn.init.xavier_uniform_(self.learnable_fill_vector) # 初始化

        # 输入维度将是 original_feature_dim + 1 (因为拼接了掩码)
        actual_in_channels = original_feature_dim + 1
        
        self.convs = nn.ModuleList()
        if use_layer_norm:
            self.norms = nn.ModuleList()
        
        # 第一层: actual_in_channels -> hidden_channels[0]
        self.convs.append(
            TransformerConv(
                actual_in_channels, # 修改这里
                hidden_channels[0] // heads,
                heads=heads,
                dropout=dropout,
                beta=True
            )
        )
        if use_layer_norm:
            self.norms.append(nn.LayerNorm(hidden_channels[0]))
        
        # 中间层: 在hidden_channels之间过渡
        for i in range(len(hidden_channels)-1):
            self.convs.append(
                TransformerConv(
                    hidden_channels[i],
                    hidden_channels[i+1] // heads,
                    heads=heads,
                    dropout=dropout,
                    beta=True
                )
            )
            if use_layer_norm:
                self.norms.append(nn.LayerNorm(hidden_channels[i+1]))
            
    def forward(self, x_orig, edge_index, missing_mask_tensor):
        # x_orig: 原始特征 (N, original_feature_dim)，缺失部分为0
        # missing_mask_tensor: (N, 1)，1表示缺失，0表示存在

        # 1. 应用学习型填充
        processed_x = x_orig.clone() # 创建副本以避免修改原始数据
        
        # 找到缺失embedding的基因的索引
        # missing_mask_tensor.squeeze() 将 (N,1) -> (N,)
        # .nonzero(as_tuple=True) 返回一个元组，第一个元素是行索引
        # [0] 取出行索引张量
        # mask_indices = (missing_mask_tensor.squeeze() == 1).nonzero(as_tuple=True)[0]
        
        # 使用 PyTorch 1.10+ 的方式获取布尔索引
        mask_indices_bool = missing_mask_tensor.squeeze().bool()


        if mask_indices_bool.any(): # 只有当存在缺失值时才进行填充
             # 将可学习向量扩展到所有缺失基因的数量
             fill_values = self.learnable_fill_vector.expand(mask_indices_bool.sum().item(), -1)
             processed_x[mask_indices_bool] = fill_values
        
        # 2. 拼接特征和掩码
        # processed_x: (N, original_feature_dim)
        # missing_mask_tensor: (N, 1)
        x_combined = torch.cat([processed_x, missing_mask_tensor], dim=-1) # (N, original_feature_dim + 1)

        # 3. 通过Transformer层
        x_out = x_combined # 重命名一下，下面用x_out
        for i in range(len(self.convs)):
            if self.residual and i > 0:
                prev_dim = x_out.size(-1)
                prev = x_out
                
            x_out = self.convs[i](x_out, edge_index)
            
            if i < len(self.convs) - 1:
                if self.use_layer_norm:
                    x_out = self.norms[i](x_out)
                x_out = F.relu(x_out)
                x_out = F.dropout(x_out, p=0.1, training=self.training) # 注意这里dropout p值固定为0.1
                
                if self.residual and i > 0 and prev_dim == x_out.size(-1):
                    x_out = x_out + prev
            else:
                if self.use_layer_norm:
                    x_out = self.norms[i](x_out)
                        
        return x_out


# 链接预测器
class LinkPredictor(nn.Module):
    def __init__(self, 
                 in_channels,        
                 hidden_channels,    # 现在接受一个列表
                 dropout=0.1):
        super().__init__()
        
        self.lins = nn.ModuleList()
        self.norms = nn.ModuleList()
        
        # 第一层: in_channels*2 -> hidden_channels[0]
        self.lins.append(nn.Linear(in_channels * 2, hidden_channels[0]))
        self.norms.append(nn.LayerNorm(hidden_channels[0]))
        
        # 中间层
        for i in range(len(hidden_channels)-1):
            self.lins.append(nn.Linear(hidden_channels[i], hidden_channels[i+1]))
            self.norms.append(nn.LayerNorm(hidden_channels[i+1]))
        
        # 最后一层: hidden_channels[-1] -> 1
        self.lins.append(nn.Linear(hidden_channels[-1], 1))
        
        self.dropout = dropout
        
    def forward(self, z, edge_index):
        x_i = z[edge_index[0]]
        x_j = z[edge_index[1]]
        x = torch.cat([x_i, x_j], dim=-1)
        
        for i in range(len(self.lins)):
            if i < len(self.lins) - 1:  # 不是最后一层
                x = self.lins[i](x)
                x = self.norms[i](x)
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
            else:  # 最后一层
                x = self.lins[i](x)
                
        return torch.sigmoid(x)