"""
build_promptmodel script  ver： Mar 25th 19:20

"""
import timm
import torch
from .structure import *


def build_promptmodel(num_classes=2, edge_size=224, model_idx='ViT', patch_size=16,
                      Prompt_Token_num=10, VPT_type="Deep"):
    # VPT_type = "Deep" / "Shallow"

    if model_idx[0:3] == 'ViT':
        # ViT_Prompt
        import timm
        from pprint import pprint
        model_names = timm.list_models('*vit*')
        pprint(model_names)

        basic_model = timm.create_model('vit_base_patch' + str(patch_size) + '_' + str(edge_size),
                                        pretrained=True)

        model = VPT_ViT(img_size=edge_size, patch_size=patch_size, Prompt_Token_num=Prompt_Token_num,
                        VPT_type=VPT_type)

        model.load_state_dict(basic_model.state_dict(), False)
        model.New_CLS_head(num_classes)
        model.Freeze()
    else:
        print("The model is not difined in the Prompt script！！")
        return -1

    try:
        img = torch.randn(1, 3, edge_size, edge_size)
        preds = model(img)  # (1, class_number)
        print('test model output：', preds)
    except:
        print("Problem exist in the model defining process！！")
        return -1
    else:
        print('model is ready now!')
        return model
