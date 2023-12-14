# Copyright (c) Meta Platforms, Inc. and affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from sentence_transformers import SentenceTransformer, util

import sys
sys.path.append("open_clip_torch/src/")
from open_clip.factory import create_model_and_transforms, get_tokenizer
import torch
import matplotlib.pyplot as plt
import numpy as np

from BLIP.models.blip_itm import blip_itm
device = "cuda"

def plot_heatmap(similarity, labels, model_name):
    # Plotting the covariance matrix
    plt.clf()
    plt.imshow(similarity, interpolation='nearest')

    # Adding the cell values
    for i in range(similarity.shape[0]):
        for j in range(similarity.shape[1]):
            plt.text(j, i, f'{similarity[i, j]:.2f}', ha='center', va='center', color='w')

    # Adding a colorbar
    plt.colorbar()

    # Labeling and title
    plt.xticks(np.arange(len(labels)), labels, rotation='vertical')
    plt.yticks(np.arange(len(labels)), labels)

    # Adjusting x-axis label alignment
    plt.gca().set_xticks(np.arange(len(labels)) - 0.25)
    plt.gca().set_xticklabels(labels, rotation='vertical', ha='center')
    plt.title(model_name)
    # Remove the surrounding box
    plt.box(False)

    # Return handle to plot
    return plt
    
# model = 'ViT-B-32' or 'ViT-L-14'
def initialize_clip(model_name):
    clip_model, _, preprocess_val = create_model_and_transforms(
        pretrained="openai",
        model_name=model_name,
        precision="fp32",
        device = device,
        jit=True,
        output_dict=True
    )
    clip_model.eval()
    tokenizer = get_tokenizer(model_name)
    return clip_model, tokenizer

def initialize_blip(model_name="model_base_14M", ):
    model_url = f"https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/{model_name}.pth"
    blip_model = blip_itm(pretrained=model_url, image_size=224, vit='base', med_config="BLIP/configs/med_config.json").eval()
    return blip_model

def get_similarity(model, text_tokens):  
    with torch.no_grad():
        text_features = model.encode_text(text_tokens).float()
    
    text_features /= text_features.norm(dim=-1, keepdim=True)
    similarity = text_features.cpu().numpy() @ text_features.cpu().numpy().T
    
    return similarity