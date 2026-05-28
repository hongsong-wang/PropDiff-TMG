import torch
import torch.nn as nn
from network.model_utils import *
from utils.utils import default
from network.dual_encoder import TextEncoder


class UNetModel(nn.Module):
    def __init__(self,
                 image_size: int = 64,
                 base_channels: int = 64,
                 dim_mults=(1, 2, 4, 8),
                 dropout: float = 0.1,
                 num_heads: int = 1,
                 world_dims: int = 3,
                 attention_resolutions=(4, 8),
                 with_attention: bool = False,
                 verbose: bool = False,
                 tensor_condition_dim: int = 3+1,  
                 use_text_condition :bool = True,
                 use_tensor_condition: bool = True,
                 ):
        super().__init__()
        self.use_tensor_condition = use_tensor_condition
        self.use_text_condition = use_text_condition
        self.tensor_condition_dim = tensor_condition_dim
        channels = [base_channels, *
                    map(lambda m: base_channels * m, dim_mults)]
        in_out = list(zip(channels[:-1], channels[1:]))

        self.verbose = verbose
        emb_dim = base_channels * 4

        # 时间步长位置嵌入
        self.time_pos_emb = LearnedSinusoidalPosEmb(base_channels)
        # 时间步长编码
        self.time_emb = nn.Sequential(
            nn.Linear(base_channels + 1, emb_dim),
            activation_function(),
            nn.Linear(emb_dim, emb_dim)
        )


        # if self.use_tensor_condition:
        #     # 条件向量编码
        #     self.tensor_emb0 = nn.Sequential(
        #         nn.Linear(base_channels + 1, emb_dim),
        #         activation_function(),
        #         nn.Linear(emb_dim, emb_dim)
        #     )
        #     self.tensor_emb1 = nn.Sequential(
        #         nn.Linear(base_channels + 1, emb_dim),
        #         activation_function(),
        #         nn.Linear(emb_dim, emb_dim)
        #     )
        #     self.tensor_emb2 = nn.Sequential(
        #         nn.Linear(base_channels + 1, emb_dim),
        #         activation_function(),
        #         nn.Linear(emb_dim, emb_dim)
        #     )
        #     # self.vol_emb = nn.Sequential(
        #     #     nn.Linear(base_channels + 1, emb_dim),
        #     #     activation_function(),
        #     #     nn.Linear(emb_dim, emb_dim)
        #     # )
        #     # 位置编码   # 可学习的正弦位置编码
        #     self.tensor_pos_emb0 = LearnedSinusoidalPosEmb(base_channels)
        #     self.tensor_pos_emb1 = LearnedSinusoidalPosEmb(base_channels)
        #     self.tensor_pos_emb2 = LearnedSinusoidalPosEmb(base_channels)
        #     # self.vol_pos_emb = LearnedSinusoidalPosEmb(base_channels)

        # if self.use_text_condition:                      # 文本嵌入编码
        #     self.text_emb = nn.Sequential(
        #         nn.Linear(text_emb_dim, emb_dim),
        #         activation_function(),
        #         nn.Linear(emb_dim, emb_dim)
        #     )
        # 加载文本编码器模型
        self.text_encoder = TextEncoder(num_projection_layers=1, input_dims=768, projection_dims=256, dropout_rate=0.1, trainable=False)
        self.text_encoder.load_state_dict(torch.load("/home/daibingxuan/workspace/microstructure_generation_3d/training_result/dual_encoder/text_encoder_new1.pth"))
        self.text_encoder.eval()  # 设置为评估模式
        
        self.input_emb = conv_nd(world_dims, 2, base_channels, 3, padding=1)
        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out)
        ds = 1

        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)
            res = image_size // ds
            use_cross = (res == 4 or res == 8)
            # use_cross = False
            self.downs.append(nn.ModuleList([
                ResnetBlock(world_dims, dim_in, dim_out,
                            emb_dim=emb_dim, dropout=dropout, use_tensor_condition=self.use_tensor_condition),
                our_Identity(),
                CrossAttention(feature_dim=dim_out, tensor_dim=emb_dim,
                               num_heads=num_heads, image_size=res, world_dims=3,
                               drop_out=dropout) if use_cross else our_Identity(),
                nn.Sequential(
                    normalization(dim_out),
                    activation_function(),
                    AttentionBlock(
                        dim_out, num_heads=num_heads)) if ds in attention_resolutions and with_attention else our_Identity(),
                Downsample(
                    dim_out, dims=world_dims) if not is_last else our_Identity()
            ]))
            if not is_last:
                ds *= 2

        mid_dim = channels[-1]
        res = image_size // ds
        self.mid_block1 = ResnetBlock(
            world_dims, mid_dim, mid_dim, emb_dim=emb_dim, dropout=dropout, use_tensor_condition=self.use_tensor_condition)
        self.mid_cross_attn = our_Identity()
        self.mid_cross_attn2 = CrossAttention(feature_dim=mid_dim, tensor_dim=emb_dim,
                                             num_heads=num_heads, image_size=res, world_dims=world_dims,
                                             drop_out=dropout) 
        self.mid_self_attn = nn.Sequential(
            normalization(mid_dim),
            activation_function(),
            AttentionBlock(mid_dim, num_heads=num_heads)
        ) if ds in attention_resolutions and with_attention else our_Identity()
        self.mid_block2 = ResnetBlock(
            world_dims, mid_dim, mid_dim, emb_dim=emb_dim, dropout=dropout, use_tensor_condition=self.use_tensor_condition)


        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            is_last = ind >= (num_resolutions - 1)
            res = image_size // ds
            use_cross = (res == 4 or res == 8)
            # use_cross = False
            self.ups.append(nn.ModuleList([
                ResnetBlock(world_dims, dim_out * 2, dim_in,
                            emb_dim=emb_dim, dropout=dropout, use_tensor_condition=self.use_tensor_condition),
                our_Identity(),
                CrossAttention(feature_dim=dim_in, tensor_dim=emb_dim,
                               num_heads=num_heads, image_size=res, world_dims=3,
                               drop_out=dropout) if use_cross else our_Identity(),
                nn.Sequential(
                    normalization(dim_in),
                    activation_function(),
                    AttentionBlock(
                        dim_in, num_heads=num_heads)) if ds in attention_resolutions and with_attention else our_Identity(),
                Upsample(
                    dim_in, dims=world_dims) if not is_last else our_Identity()
            ]))
            if not is_last:
                ds //= 2

        self.end = nn.Sequential(
            normalization(base_channels),
            activation_function()
        )

        self.out = conv_nd(world_dims, base_channels, 1, 3, padding=1)

    def forward(self, x, t=torch.ones((16)), caption=None, prop=None, x_self_cond=None): 
        
        x_self_cond = default(x_self_cond, lambda: torch.zeros_like(x))

        x = torch.cat((x, x_self_cond), dim=1)

        if self.verbose:
            print("input size:")
            print(x.shape)

        x = self.input_emb(x)
        t = self.time_emb(self.time_pos_emb(t))

        # vol_condition = None
        # if self.use_tensor_condition:
        #     tensor_condition_tmp = torch.zeros((t.shape[0], t.shape[1], self.tensor_condition_dim)).cuda()

        #     tensor_condition_tmp[:, :, 0] = self.tensor_emb0(self.tensor_pos_emb0(tensor_condition[:, 0]))
        #     tensor_condition_tmp[:, :, 1] = self.tensor_emb1(self.tensor_pos_emb1(tensor_condition[:, 1]))
        #     tensor_condition_tmp[:, :, 2] = self.tensor_emb2(self.tensor_pos_emb2(tensor_condition[:, 2]))
        #     # tensor_condition_tmp[:, :, 3] = self.vol_emb(self.vol_pos_emb(tensor_condition[:, 3]))
        #     tensor_condition = tensor_condition_tmp[:, :, :]
            # vol_condition = tensor_condition_tmp[:, :, 3:4].permute(0, 2, 1)
        if caption is None:
            text_condition = torch.zeros((x.shape[0], 256)).to(x.device) 
        else:
            text_condition = self.text_encoder(caption, prop)  # 通过文本编码器提取文本特征

        h = []

        for resnet, cross_attn, cross_attn2, self_attn, downsample in self.downs:
            x = resnet(x, t, text_condition)   # 残差网络层嵌入文本特征
            if self.verbose:
                print(x.shape)
                if type(cross_attn) == CrossAttention:
                    print("cross attention at resolution: ",
                          cross_attn.image_size)
            x = cross_attn(x)
            x = cross_attn2(x, text_condition)                   # 交叉注意力机制中融入文本特征，引导生成
            x = self_attn(x)
            if self.verbose:
                print(x.shape)
            h.append(x)
            x = downsample(x)
            if self.verbose:
                print(x.shape)

        if self.verbose:
            print("enter bottle neck")
        x = self.mid_block1(x, t, text_condition)    # 残差网络层嵌入文本特征
        if self.verbose:
            print(x.shape)

        x = self.mid_cross_attn(x)
        x = self.mid_cross_attn2(x, text_condition)                    # 交叉注意力机制中融入文本特征，引导生成
        x = self.mid_self_attn(x)
        if self.verbose:
            print("cross attention at resolution: ",
                  self.mid_cross_attn.image_size)
            print(x.shape)
        x = self.mid_block2(x, t, text_condition)    # 残差网络层嵌入文本特征
        if self.verbose:
            print(x.shape)
            print("finish bottle neck")

        for resnet, cross_attn, cross_attn2, self_attn, upsample in self.ups:
            x = torch.cat((x, h.pop()), dim=1)
            if self.verbose:
                print(x.shape)
            x = resnet(x, t, text_condition)        # 残差网络层嵌入文本特征
            if self.verbose:
                print(x.shape)
            x = cross_attn(x)
            x = cross_attn2(x, text_condition)                        # 交叉注意力机制中融入文本特征，引导生成
            x = self_attn(x)
            if self.verbose:
                if type(cross_attn) == CrossAttention:
                    print("cross attention at resolution: ",
                          cross_attn.image_size)
                print(x.shape)
            x = upsample(x)
            if self.verbose:
                print(x.shape)

        x = self.end(x)
        if self.verbose:
            print(x.shape)
            
        x=self.out(x)
       
        return x
