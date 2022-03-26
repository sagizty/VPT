"""
Unofficial code for VPT(Visual Prompt Tuning) paper of arxiv 2203.12119

A toy Tuning process that demostrates the code

the code is based on timm

"""
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from PromptModels.GetPromptModel import build_promptmodel


def setup_seed(seed):  # setting up the random seed
    import random
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


setup_seed(42)
batch_size=2
edge_size=384
data = torch.randn(batch_size, 3, edge_size, edge_size)
labels = torch.ones(batch_size).long()  # long ones

model = build_promptmodel(num_classes=3, edge_size=edge_size, model_idx='ViT', patch_size=16,
                          Prompt_Token_num=10, VPT_type="Deep")  # VPT_type = "Shallow"

optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=0.01)
criterion = nn.CrossEntropyLoss()

preds = model(data)  # (1, class_number)
# print('before Tuning model output：', preds)

# check backwarding tokens
for param in model.parameters():
    if param.requires_grad:
        print(param.shape)

for i in range(3):
    print('epoch:',i)
    optimizer.zero_grad()
    outputs = model(data)
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()

preds = model(data)  # (1, class_number)
print('After Tuning model output：', preds)