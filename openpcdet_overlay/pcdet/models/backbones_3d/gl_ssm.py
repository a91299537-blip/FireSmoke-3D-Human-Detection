import torch
import torch.nn as nn
import torch.nn.functional as F
from mamba_ssm import Mamba  
from torch.nn.utils.rnn import pad_sequence

# === 1. 定义直通估计器 (STE) 解决 round 不可导问题 ===
class RoundSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return torch.round(x)
        
    @staticmethod
    def backward(ctx, grad_output):
        # 梯度直接无损穿透
        return grad_output, None 

# === 2. 真正的分布感知量化器 ===
class MambaQuantizer(nn.Module):
    def __init__(self, bits=4):
        super().__init__()
        self.bits = bits
        # 设置为可学习的参数，让网络在 QAT 阶段自己微调缩放因子
        self.scale = nn.Parameter(torch.tensor(1.0))
        self.q_max = 2**(self.bits - 1) - 1
        self.q_min = -2**(self.bits - 1)

    def forward(self, x):
        if not self.training and self.bits == 16:
            return x
            
        # 核心：使用 STE 替代原生的 torch.round
        x_scaled = x / self.scale
        x_quant = RoundSTE.apply(x_scaled).clamp(self.q_min, self.q_max)
        
        # 反量化回浮点数参与后续矩阵乘法
        return x_quant * self.scale

# ==========================================
# 真·MambaQuant：方差对齐 + 前景感知量化
# ==========================================
class MambaQuantizer(nn.Module):
    def __init__(self, dim, bits=4, foreground_thresh=0.5):
        super().__init__()
        self.bits = bits
        self.foreground_thresh = foreground_thresh
        
        # 🌟 创新点一：方差对齐 (Variance Alignment)
        # 不再是标量，而是通道级的可学习参数 (1, 1, Dim)，拉齐不同特征维度的分布
        self.scale = nn.Parameter(torch.ones(1, 1, dim))
        
        self.q_max = 2**(self.bits - 1) - 1
        self.q_min = -2**(self.bits - 1)

    def forward(self, x, gate=None):
        if self.bits == 16:
            return x
            
        # 1. 执行通道级方差对齐的量化
        scale = self.scale.abs().clamp_min(1e-6)
        x_scaled = x / scale
        x_quant = RoundSTE.apply(x_scaled).clamp(self.q_min, self.q_max)
        x_dequant = x_quant * scale
        
        # 🌟 创新点二：前景感知低位宽阻断 (Foreground-Aware Blocking)
        # 🌟 创新点二：前景感知低位宽阻断 (Foreground-Aware Blocking)
        if gate is not None:
            # 【修复】：将维度维度的 gate 取平均，得到每个点(Token)真实的全局前景概率
            # token_gate 形状为 (1, Length, 1)
            token_gate = gate.mean(dim=-1, keepdim=True)
            
            # 使用点级别概率生成掩码，整个点统一阻断或量化
            foreground_mask = (token_gate > self.foreground_thresh).float()
            
            # 前景使用无损的原输入 x，背景使用量化后的 x_dequant
            out = x * foreground_mask + x_dequant * (1 - foreground_mask)
        else:
            out = x_dequant
            
        return out

# ==========================================
# 1. 局部乘法聚合模块 (LMA)
# ==========================================
class LocalMultiplicativeAggregation(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gate_conv = nn.Conv1d(dim, dim, kernel_size=3, padding=1, groups=dim)
        self.linear = nn.Linear(dim, dim)
        self.activation = nn.SiLU()

    def forward(self, x):
        x_t = x.transpose(1, 2) 
        gate = self.activation(self.gate_conv(x_t)).transpose(1, 2)
        out = self.linear(x * gate) 
        # 🌟 核心修改：不仅返回特征，还把 gate (前景概率) 吐出来
        return out, gate

# ==========================================
# 2. 双尺度状态空间模块 (DSB)
# ==========================================
class DualScaleSSMBlock(nn.Module):
    def __init__(self, dim, d_state=16, expand=2, use_dual_scale=True, bits=16):
        super().__init__()
        self.use_dual_scale = use_dual_scale
        self.bits = bits
        
        self.mamba_forward = Mamba(d_model=dim, d_state=d_state, expand=expand)
        self.mamba_backward = Mamba(d_model=dim, d_state=d_state, expand=expand)
        
        if self.use_dual_scale:
            self.downsample = nn.Conv1d(dim, dim, kernel_size=2, stride=2)
            self.upsample = nn.ConvTranspose1d(dim, dim, kernel_size=2, stride=2)
            
        if self.bits < 16:
            # 传入 dim 以支持方差对齐
            self.quantizer = MambaQuantizer(dim=dim, bits=self.bits)

    # 🌟 接收来自 LMA 的前景门控信号
    def forward(self, x_xyz, x_zyx=None, gate_xyz=None, gate_zyx=None):
        if self.bits < 16:
            # 将特征和前景信号一起送入量化器
            x_xyz = self.quantizer(x_xyz, gate_xyz)
            if x_zyx is not None:
                x_zyx = self.quantizer(x_zyx, gate_zyx)
                
        out_xyz = self.mamba_forward(x_xyz)
        if x_zyx is None:
            return out_xyz + x_xyz, None
            
        if self.use_dual_scale:
            x_zyx_down = self.downsample(x_zyx.transpose(1, 2)).transpose(1, 2)
            out_zyx_down = self.mamba_backward(x_zyx_down.flip(dims=[1])) 
            out_zyx = self.upsample(out_zyx_down.transpose(1, 2)).transpose(1, 2).flip(dims=[1])
            diff = x_zyx.shape[1] - out_zyx.shape[1]
            if diff > 0:
                out_zyx = F.pad(out_zyx, (0, 0, 0, diff))
            elif diff < 0:
                out_zyx = out_zyx[:, :x_zyx.shape[1], :]
        else:
            out_zyx = self.mamba_backward(x_zyx.flip(dims=[1])).flip(dims=[1])
            
        return out_xyz + x_xyz, out_zyx + x_zyx

# ==========================================
# 3. 核心骨干网 (支持动态无垫零与消融控制)
# ==========================================
class GLSSMBackbone(nn.Module):
    def __init__(self, model_cfg, input_channels, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg
        
        self.dim = self.model_cfg.DIM
        self.num_layers = self.model_cfg.NUM_LAYERS
        self.d_state = self.model_cfg.D_STATE
        self.expand = self.model_cfg.EXPAND
        self.bits = self.model_cfg.get('BITS', 16)
        self.num_point_features = self.model_cfg.CONV_OUT_CHANNEL
        
        # 读取消融开关 (默认全开以保证兼容性)
        self.use_lma = getattr(self.model_cfg, 'USE_LMA', True)
        self.use_dual_path = getattr(self.model_cfg, 'USE_DUAL_PATH', True)
        self.use_dual_scale = getattr(self.model_cfg, 'USE_DUAL_SCALE', True)

        if self.use_lma:
            self.lma = LocalMultiplicativeAggregation(self.dim)
            
        self.blocks = nn.ModuleList([
            DualScaleSSMBlock(
                self.dim,
                self.d_state,
                self.expand,
                self.use_dual_scale,
                bits=self.bits,
            )
            for _ in range(self.num_layers)
        ])

    def compute_morton_code_3d(self, coords, order='xyz'):
        batch_idx, z, y, x = coords[:, 0].long(), coords[:, 1].long(), coords[:, 2].long(), coords[:, 3].long()
        if order == 'xyz':
            v1, v2, v3 = x, y, z
        else:  
            v1, v2, v3 = z, y, x
        
        answer = torch.zeros_like(x)
        for i in range(10):
            answer |= ((v1 & (1 << i)) << (2 * i)) | \
                  ((v2 & (1 << i)) << (2 * i + 1)) | \
                  ((v3 & (1 << i)) << (2 * i + 2))
        return answer

    def group_free_serialization(self, voxel_coords, voxel_features):
        batch_idx = voxel_coords[:, 0].long()
        
        morton_xyz = self.compute_morton_code_3d(voxel_coords, order='xyz')
        sort_keys_xyz = batch_idx * (morton_xyz.max() + 1) + morton_xyz
        sort_idx_xyz = torch.argsort(sort_keys_xyz)
        
        sort_idx_zyx = None
        if self.use_dual_path:
            morton_zyx = self.compute_morton_code_3d(voxel_coords, order='zyx')
            sort_keys_zyx = batch_idx * (morton_zyx.max() + 1) + morton_zyx
            sort_idx_zyx = torch.argsort(sort_keys_zyx)
    
        return sort_idx_xyz, sort_idx_zyx

    def forward(self, batch_dict):
        voxel_features = batch_dict['voxel_features']
        voxel_coords = batch_dict['voxel_coords'] 
        batch_size = batch_dict['batch_size']

        # === 加入这两行防御性代码 ===
        if voxel_features.shape[0] == 0:
            return batch_dict
        # ===========================

        idx_xyz, idx_zyx = self.group_free_serialization(voxel_coords, voxel_features)

        feat_xyz = voxel_features[idx_xyz]
        batch_idx_xyz = voxel_coords[idx_xyz][:, 0]

        if self.use_dual_path:
            feat_zyx = voxel_features[idx_zyx]
            batch_idx_zyx = voxel_coords[idx_zyx][:, 0]

        out_xyz_list = []
        out_zyx_list = []

        # 动态无垫零处理
        for i in range(batch_size):
            # 提取特征
            x_xyz_i = feat_xyz[batch_idx_xyz == i].unsqueeze(0)
            if x_xyz_i.shape[1] == 0:
                continue
            x_zyx_i = feat_zyx[batch_idx_zyx == i].unsqueeze(0) if self.use_dual_path else None

            # 🌟 接收特征和前景 Gate
            gate_xyz_i, gate_zyx_i = None, None
            if self.use_lma:
                x_xyz_i, gate_xyz_i = self.lma(x_xyz_i)
                if x_zyx_i is not None:
                    x_zyx_i, gate_zyx_i = self.lma(x_zyx_i)
                
            # 🌟 将 Gate 传给后方的 Mamba 模块进行前景阻断
            for block in self.blocks:
                x_xyz_i, x_zyx_i = block(x_xyz_i, x_zyx_i, gate_xyz=gate_xyz_i, gate_zyx=gate_zyx_i)

            out_xyz_list.append(x_xyz_i.squeeze(0))
            if x_zyx_i is not None:
                out_zyx_list.append(x_zyx_i.squeeze(0))

        processed_xyz = torch.cat(out_xyz_list, dim=0)
        final_features = torch.zeros_like(voxel_features)
        final_features[idx_xyz] = processed_xyz

        if self.use_dual_path:
            processed_zyx = torch.cat(out_zyx_list, dim=0)
            final_zyx = torch.zeros_like(voxel_features)
            final_zyx[idx_zyx] = processed_zyx
            # 融合
            final_features = final_features + final_zyx

        batch_dict['voxel_features'] = final_features
        return batch_dict
