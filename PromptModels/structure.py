"""
VPT script  ver： Mar 25th 19:20

"""
import timm

import torch
import torch.nn as nn

from timm.models.vision_transformer import VisionTransformer, PatchEmbed


class VPT_ViT(VisionTransformer):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dim=768, depth=12,
                 num_heads=8, mlp_ratio=4., qkv_bias=True, representation_size=None, distilled=False,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0., embed_layer=PatchEmbed, norm_layer=None,
                 act_layer=None, weight_init='', Prompt_Token_num=1, VPT_type="Shallow"):

        # Recreate ViT
        super().__init__(img_size, patch_size, in_chans, num_classes, embed_dim, depth, num_heads, mlp_ratio, qkv_bias,
                         representation_size, distilled, drop_rate, attn_drop_rate, drop_path_rate, embed_layer,
                         norm_layer, act_layer, weight_init)
        self.VPT_type = VPT_type
        if VPT_type == "Deep":
            self.Prompt_Tokens = nn.Parameter(torch.zeros(depth, Prompt_Token_num, embed_dim))
        else:  # "Shallow"
            self.Prompt_Tokens = nn.Parameter(torch.zeros(1, Prompt_Token_num, embed_dim))

    def New_CLS_head(self, new_classes=15):
        self.head = nn.Linear(self.embed_dim, new_classes)

    def Freeze(self):
        for param in self.parameters():
            param.requires_grad = False

        self.Prompt_Tokens.requires_grad = True
        for param in self.head.parameters():
            param.requires_grad = True

    def forward_features(self, x):
        x = self.patch_embed(x)
        # print(x.shape,self.pos_embed.shape)
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)

        # concatenate CLS token
        x = torch.cat((cls_token, x), dim=1)
        x = self.pos_drop(x + self.pos_embed)

        if self.VPT_type == "Deep":

            Prompt_Token_num = self.Prompt_Tokens.shape[1]

            for i in range(len(self.blocks)):
                # concatenate Prompt_Tokens
                Prompt_Tokens = self.Prompt_Tokens[i].unsqueeze(0)
                x = torch.cat((x, Prompt_Tokens.expand(x.shape[0], -1, -1)), dim=1)
                num_tokens = x.shape[1]
                x = self.blocks[i](x)[:, :num_tokens - Prompt_Token_num]

        else:  # self.VPT_type == "Shallow"
            # concatenate Prompt_Tokens
            Prompt_Tokens = self.Prompt_Tokens.expand(x.shape[0], -1, -1)
            x = torch.cat((x, Prompt_Tokens), dim=1)
            # Sequntially procees
            x = self.blocks(x)

        x = self.norm(x)
        return self.pre_logits(x[:, 0])  # use cls token for cls head

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x
