# Copyright (c) Meta Platforms, Inc. and affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import os
import webdataset as wds
import torch
from PIL import Image, ImageOps
import matplotlib.pyplot as plt
import torch.nn as nn
import argparse
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from sentence_transformers import SentenceTransformer, util
from BLIP.models.blip import blip_decoder
import sys
sys.path.append("open_clip_torch/src/")
from training.data import (log_and_continue,
                             get_dataset_size,
                             tarfile_to_samples_nothrow,
                             filter_no_caption_or_no_image)
from open_clip.factory import create_model_and_transforms, get_tokenizer
### packages
import numpy as np
import distutils
import distutils.util
   
def inference_func(args):
    # models
    # initialize models
    CAPTIONING = args.captioning
    CLIPING = args.cliping
    CLIPCAP = args.clipcap
    assert CAPTIONING or CLIPING or CLIPCAP 
    if CAPTIONING or CLIPCAP:
        assert args.model_url is not None, ("Please provide captioning path, model_url={} \n"
                                            "for BLIP w/ ViT-B pretrained on 14M image-text pairs use:\n" 
                                                "https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model_base_14M.pth \n"
                                            "for BLIP w/ ViT-B pretrained on 129M image-text pairs use:\n" 
                                                "https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model_base.pth".format(args.model_url))
    model_size = args.clip_model_size
    bs_size = args.per_device_batch_size
    num_workers = args.num_workers
    input_shards = args.data_dir
    assert os.path.exists(os.path.dirname(input_shards)), f"Parent directory does not exist: {os.path.dirname(input_shards)}"
    num_samples, num_shards = get_dataset_size(input_shards)
    print(f"Num of Shards {num_shards} - Num of Samples {num_samples}")
    device = torch.device(args.device)
    
    if CLIPING:
        if model_size == 'base':
            clip_model_name = 'ViT-B-32'
        elif model_size == 'large':
            clip_model_name = 'ViT-L-14'
        else:
            raise NotImplementedError
    else:
        clip_model_name = 'ViT-B-32' # create a small model just to use preprocessing of datacomp
    
    if CLIPCAP:  # by default use large model if both cliping and captioning model is to be used
        clip_model_name = 'ViT-L-14'
        
    # need for preprocessing function in openclip
    clip_model, _, preprocess_val = create_model_and_transforms(
        pretrained="openai",
        model_name=clip_model_name,
        precision="fp32",
        device = device,
        jit=True,
        output_dict=True
    )
    
    if CLIPING or CLIPCAP:
        tokenizer = get_tokenizer(clip_model_name)
        clip_model.eval()

        if args.distributed and args.sync_bn:
            clip_model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(clip_model)

        clip_model_without_ddp = clip_model
        if args.distributed:
            clip_model = torch.nn.parallel.DistributedDataParallel(clip_model, device_ids=[args.gpu])
            clip_model_without_ddp = clip_model.module
    
    if CAPTIONING or CLIPCAP:
        captioning_model_size = "base"     
        blip_model = blip_decoder(pretrained=args.model_url, image_size=224, vit=f'{captioning_model_size}').to(device)
        
        if args.sentence_language_model == 'multilingual':
            # 135 Million parameters
            sen_model = SentenceTransformer('sentence-transformers/distiluse-base-multilingual-cased-v1').to(device)
        elif args.sentence_language_model == 'unilingual':
            # 22 Million parameters
            sen_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2').to(device)
        
        print(f'Sentence embedding model: {args.sentence_language_model}')
    
        blip_model.eval()
        sen_model.eval()

        if args.distributed and args.sync_bn:
            blip_model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(blip_model)
            sen_model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(sen_model)

        blip_model_without_ddp = blip_model
        sen_model_without_ddp = sen_model
        if args.distributed:
            blip_model = torch.nn.parallel.DistributedDataParallel(blip_model, device_ids=[args.gpu])
            blip_model_without_ddp = blip_model.module
            sen_model = torch.nn.parallel.DistributedDataParallel(sen_model, device_ids=[args.gpu])
            sen_model_without_ddp = sen_model.module

    pipeline = [wds.SimpleShardList(args.data_dir)]
    pipeline.extend([wds.split_by_node])
    pipeline.extend([wds.split_by_worker])
    pipeline.extend([tarfile_to_samples_nothrow])
    pipeline.extend([
        wds.select(filter_no_caption_or_no_image),
        wds.decode("pilrgb", handler=log_and_continue),
        wds.rename(image="jpg;png;jpeg;webp", text="txt", uid="json", original_width="json", original_height="json"),
        wds.map_dict(image=preprocess_val, 
                     uid=lambda data: data["uid"], 
                     original_width=lambda data: data["original_width"], 
                     original_height=lambda data: data["original_height"]),
        wds.to_tuple("__key__", "uid", "image", "text", "original_width", "original_height"),
        wds.batched(args.per_device_batch_size, partial=True)
    ])
    dataset = wds.DataPipeline(*pipeline)
    dataloader = wds.WebLoader(
        dataset,
        batch_size=None,
        shuffle=False,
        num_workers=num_workers,
        persistent_workers=num_workers > 0,
    )

    print(f'Number of GPUS: {torch.cuda.device_count()}')
    print(f'Batch size: {bs_size}')
    
    if args.clipcap:
        model_name = f'CLIPCAP: CLIP {clip_model_name} BLIP-base'
    else:
        model_name = f'BLIP-base' if CAPTIONING else f'CLIP_{model_size}'

    print(f"Model name is: {model_name}")
    cap_bestscores_list = []
    cap_meanscores_list = []
    clip_scores_list = []
    key_list = []
    uid_list = []
    original_width_list = []
    original_height_list = []
    generated_caption_list = []
    all_generated_caption_list = []
    caption_list = []
    with torch.no_grad():
        for idx, sample in enumerate(dataloader):
            images = sample[2]
            print(f"Current batch {idx}") 
            captions = sample[3]
            keys = sample[0]
            uids = sample[1]
            orig_width = sample[4]
            orig_height = sample[5]
            num_sequences_to_generate = 8
            if CAPTIONING or CLIPCAP:
                generated_caption = blip_model_without_ddp.generate(images.to(device), sample=True, 
                                                            top_p=0.9, max_length=20, min_length=5) 

                ## caption similarity
                #Compute embedding for both lists
                if args.use_clip_text_encoder:
                    assert CLIPCAP
                    # tokenize  and then embed true captions
                    tokenized_captions = torch.stack([tokenizer(cap)[0] for cap in captions], dim=0)
                    tokenized_captions = tokenized_captions.to(device)
                    embedding_cap = clip_model_without_ddp.encode_text(tokenized_captions, normalize=True)
                    # tokenize  and then embed generated captions
                    tokenized_gen_captions = torch.stack([tokenizer(cap)[0] for cap in generated_caption], dim=0)
                    tokenized_gen_captions = tokenized_gen_captions.to(device)
                    embedding_gen = clip_model_without_ddp.encode_text(tokenized_gen_captions, normalize=True)
                else: # use sentence transformer
                    embedding_cap = sen_model_without_ddp.encode(captions, convert_to_tensor=True)
                    embedding_gen = sen_model_without_ddp.encode(generated_caption, convert_to_tensor=True)
                
                # embeddings from sentence transformer already normalized
                current_batch_size = len(captions)
                num_features = embedding_cap.shape[1]
                ecap_norm = embedding_cap.view(current_batch_size, 1, num_features)
                egen_norm = embedding_gen.view(current_batch_size, num_sequences_to_generate, num_features)
                cosine_sim = torch.matmul(ecap_norm, egen_norm.transpose(2,1)).squeeze(dim=1)
                cap_best_scores = torch.amax(cosine_sim, dim=1) # save best scores
                cap_mean_scores = torch.mean(cosine_sim, dim=1) 
            
                if num_sequences_to_generate == 1: # only save captioning if one caption is generated
                    generated_caption_list.extend(generated_caption)
                else: # save caption with highest score
                    num_rows, num_cols = cosine_sim.shape
                    col_best_score = torch.argmax(cosine_sim, dim=1).cpu().numpy()
                    row_best_score = torch.arange(num_rows)
                    # map 2d to 1d
                    index_1d = ((row_best_score * num_cols) + col_best_score).tolist()
                    best_generated_captions = [generated_caption[idx] for idx in index_1d]
                    # save best captions
                    generated_caption_list.extend(best_generated_captions)
                    if args.save_all_captions:
                        # rearrage list of captions from a list of length B*num_sequences_to_generate to list of lists
                        # each lists consists of generated captions for each image
                        generated_caption_2d = [generated_caption[i:i+num_cols] for i in range(0, len(generated_caption), num_cols)]
                        all_generated_caption_list.extend(generated_caption_2d)
            
            # save openclip scores
            if CLIPING or CLIPCAP:
                tokenized_captions = torch.stack([tokenizer(cap)[0] for cap in captions], dim=0)
                img = images.to(device)
                txt = tokenized_captions.to(device)
                img_f = clip_model_without_ddp.encode_image(img)
                txt_f = clip_model_without_ddp.encode_text(txt)
                img_f = img_f / img_f.norm(dim=-1, keepdim=True)
                txt_f = txt_f / txt_f.norm(dim=-1, keepdim=True)
                clip_current_scores = torch.diag(img_f @ txt_f.T)
            # save scores
            if CAPTIONING or CLIPCAP:
                cap_bestscores_list.extend(cap_best_scores.cpu().numpy().tolist())
                cap_meanscores_list.extend(cap_mean_scores.cpu().numpy().tolist())
            if CLIPING or CLIPCAP:
                clip_scores_list.extend(clip_current_scores.cpu().numpy().tolist())
            
            # save identifiers
            caption_list.extend(captions)
            uid_list.extend(uids)
            key_list.extend(keys)
            original_width_list.extend(orig_width.tolist())
            original_height_list.extend(orig_height.tolist())

        result = {'uid': uid_list, 
                'key': key_list,
                'original_width':original_width_list, 
                'original_height':original_height_list,
                'text': caption_list, 
                'generatedtext': generated_caption_list, # can be empty for clip models
                'all_generated_caption_list': all_generated_caption_list, # save all generated captions from VLM text decoder
                'clip_scores': clip_scores_list,
                'cap_scores': cap_bestscores_list,
                'cap_mean': cap_meanscores_list}
        
        # remove empty fields
        if len(result['generatedtext']) == 0:
            del result['generatedtext']
        if len(result['clip_scores']) == 0:
            del result['clip_scores']
        if len(result['cap_scores']) == 0:
            del result['cap_scores']
        if len(result['cap_mean']) == 0:
            del result['cap_mean']
        if len(result['all_generated_caption_list']) == 0:
            del result['all_generated_caption_list']

        return result

def write_results(args, data):
    # Convert the dictionary to a Pandas DataFrame
    df = pd.DataFrame.from_dict(data)
    # Convert the DataFrame to a PyArrow Table
    table = pa.Table.from_pandas(df)
    # Specify the file path to save the Parquet file
    if args.clipcap:
        model_name = f'CLIPCAP'
    else:
        model_name = f'Captioning_base' if args.captioning else f'Cliping_{args.clip_model_size}'
    num_samples = len(data['uid'])
    file_pattern = f'{args.rank}_{model_name}_{num_samples}.parquet'
    # Write the Table to Parquet
    pq.write_table(table, os.path.join(args.output_dir,file_pattern)) 


def init_distributed_mode(args):
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ["WORLD_SIZE"])
        args.gpu = int(os.environ["LOCAL_RANK"])
    elif "SLURM_PROCID" in os.environ:
        args.rank = int(os.environ["SLURM_PROCID"])
        args.gpu = args.rank % torch.cuda.device_count()
    elif hasattr(args, "rank"):
        pass
    else:
        print("Not using distributed mode")
        args.distributed = False
        return

    args.distributed = True

    torch.cuda.set_device(args.gpu)
    args.dist_backend = "nccl"
    print(f"| distributed init (rank {args.rank}): {args.dist_url}", flush=True)
    torch.distributed.init_process_group(
        backend=args.dist_backend, init_method=args.dist_url, world_size=args.world_size, rank=args.rank
    )
    torch.distributed.barrier()
    if not(args.verbose):
        setup_for_distributed(args.rank == 0)

def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__

    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop("force", False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print
            
def main():
    parser = argparse.ArgumentParser(description='Description of your program.')
    # Add arguments to the parser
    parser.add_argument('--data_dir', default="/checkpoint/nasmahmoud/datacomp_data/small/shards/{00000000..00000001}.tar",
                        type=str, help='Path to shard files in .tar extension')
    parser.add_argument('--num_workers', default=10, type=int, help='Number of workers')
    parser.add_argument('--per_device_batch_size', default= 16, type=int, help='Global inference batch size')
    parser.add_argument('--captioning', action='store_true', help='Run captioning model')
    parser.add_argument('--model_url', help='Provide pretrained path to captioning model')
    parser.add_argument('--cliping', action='store_true', help='Run clip model')
    parser.add_argument('--clipcap', action='store_true', help='Run cap + clip model') # will always choose large clip
    parser.add_argument('--save_all_captions', action='store_true', 
                        help='save all generated captions from VLM decoder')
    parser.add_argument('--clip_model_size', default='base', type=str, choices=['base', 'large'], 
                        help='Choose ViT model size')
    parser.add_argument('--sentence_language_model', default='unilingual', type=str, choices=['unilingual', 'multilingual'], 
                        help='choose either a unilingual or a multilingual model for text similarity')
    parser.add_argument('--use_clip_text_encoder', action='store_true', 
                        help='When measuring sentence similarity, use clip text encoder to embedd caption and generated caption')
    
    parser.add_argument("--device", default="cuda", type=str, help="device (Use cuda or cpu Default: cuda)")
    parser.add_argument("--world-size", default=1, type=int, help="number of distributed processes")
    parser.add_argument("--dist-url", default="env://", type=str, help="url used to set up distributed training")
    parser.add_argument(
        "--sync-bn",
        dest="sync_bn",
        help="Use sync batch norm",
        action="store_true",
    )
    parser.add_argument("--verbose",help="print what each process is seeing", action="store_true")
    parser.add_argument('--output_dir', default="inference_results/", type=str)
    
    # Parse the command-line arguments
    args = parser.parse_args()
    init_distributed_mode(args)
    print(args)
    result_dict = inference_func(args)
    # save results 
    write_results(args, data=result_dict)

if __name__ == '__main__':
    main()
            