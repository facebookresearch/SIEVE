# Copyright (c) Meta Platforms, Inc. and affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import fsspec
from sentence_transformers import SentenceTransformer
import torch
import torch.nn
from torch.utils.data import Dataset, DataLoader
import argparse
import time
import os
import glob
import multiprocessing
from multiprocessing import Pool, current_process, Queue
from tqdm import tqdm
import datetime
import re
import sys
sys.path.append("open_clip_torch/src/")
from open_clip.factory import create_model_and_transforms, get_tokenizer


queue = Queue()
NUM_CAPTIONS = 8

REMOVE_PHRASES_LONG = ["an image of", "a photo of", "an icon of", "an illustration of",
                                                "a template of", "a thumbnail of", "a vector of", 
                                                 "photo stock", "stock photo", 
                                                 "a photo", "an image", "an icon", "an illustration", "a template", "a thumbnail",
                                                 "image", "photo", "icon", "illustration", "template", "vector", "thumbnail",
                                                 "free", "print", "sale", "quot", "png", "jpeg", "jpg"]

REMOVE_PHRASES = ['an image of', "a photo of", "stock photo", "photo stock", "a photo", "an image", "image", "photo"]

print(f'The list of phrases to exclude from captions and generated captions: {REMOVE_PHRASES}')
   
def remove_phrases(sentences, phrases_to_remove=REMOVE_PHRASES):
    modified_sentences = []
    # ignore case sensitivity
    phrases_to_remove = [re.compile(phrase, re.IGNORECASE) for phrase in phrases_to_remove]
    for sentence in sentences:
        for phrase in phrases_to_remove:
            sentence = phrase.sub('', sentence)
        modified_sentences.append(sentence)
    return modified_sentences

def read_parquet(filename):
    fs, url = fsspec.core.url_to_fs(filename)
    df = pd.read_parquet(url, columns=["uid", "key", "text", "generatedtext", 
                                       "all_generated_caption_list", "cap_scores"], filesystem=fs)
    uids = df["uid"].tolist()
    keys = df['key'].tolist()
    raw_captions = df["text"].tolist()
    captions = remove_phrases(raw_captions)
    best_generated_text = df["generatedtext"].tolist()
    generated_text_list = df["all_generated_caption_list"].tolist()
    generated_text_list = [remove_phrases(array.tolist())[0:NUM_CAPTIONS] for array in generated_text_list]
    cap_scores  = df['cap_scores'].tolist()
    return uids, keys, captions, generated_text_list, best_generated_text, cap_scores, raw_captions

def inference_func(inputs):
    # path to output file
    args, parquet_file = inputs  
    print(f"Processing: {parquet_file}") 
    output_file_path = os.path.join(args.output_dir, f'{os.path.basename(parquet_file)}')
    if os.path.isfile(output_file_path):
        print(f"File {output_file_path} exists!")
        return
    else:
        print("File does not exist and will be processed")
    
    gpu_id = queue.get()
    try:
        device_to_use = f'cuda:{gpu_id}'
        # get model
        if args.sentence_transformer_text_encoder:
            print('Use Sentence Transformer for encoding captions')
            if args.sentence_language_model == 'multilingual':
                # 135 Million parameters
                embed_model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2', device=device_to_use) # 117Million
            elif args.sentence_language_model == 'unilingual':
                # 22 Million parameters
                embed_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2', device=device_to_use) # 22 Million
        elif args.clip_text_encoder:
            print('Use CLIP text encoder for encoding captions')
            # need for preprocessing function in openai
            clip_model_name = 'ViT-L-14'
            embed_model, _, preprocess_val = create_model_and_transforms(
                pretrained="openai",
                model_name=clip_model_name,
                precision="fp32",
                device = device_to_use,
                jit=False,
                output_dict=True
            )
            del embed_model.visual
            clip_tokenizer = get_tokenizer(clip_model_name)
        
        elif args.blip_text_encoder:
            print('Use BLIP text encoder for encoding captions')
            from BLIP.models.blip_itm import blip_itm
            IMAGE_SIZE = 224
            model_url = 'https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model_base_14M.pth'
            embed_model = blip_itm(pretrained=model_url, image_size=IMAGE_SIZE, vit='base').to(device_to_use)
            del embed_model.visual_encoder
            
        embed_model.eval()
            
        # read data
        uids, keys, captions, generated_text_list, best_generated_text, cap_scores, raw_captions = read_parquet(parquet_file)
        batch_size = args.per_device_batch_size
        number_samples = len(captions)
        num_gen_captions = NUM_CAPTIONS
        print(f'num of samples {number_samples}')
        
        best_generated_caption_list = []
        caption_list = []
        uid_list = []
        keys_list = []
        new_scores = []
        old_scores = []
        print('Start Inference')
        with torch.no_grad():
            for start_idx in tqdm(range(0, len(captions), batch_size)):
                # save raw captions
                curr_raw_captions = raw_captions[start_idx:start_idx+batch_size]
                curr_captions = captions[start_idx:start_idx+batch_size]
                curr_bt_size = len(curr_captions) # last batch size fix
                curr_generated_captions = []
                for gen in generated_text_list[start_idx:start_idx+batch_size]:
                    curr_generated_captions.extend(gen)
                # join
                all_captions = []
                all_captions.extend(curr_captions)
                all_captions.extend(curr_generated_captions)
                # compute embeddings
                if args.sentence_transformer_text_encoder:
                    embeddings = embed_model.encode(all_captions, convert_to_tensor=True, batch_size=batch_size)
                elif args.clip_text_encoder:
                    tokenized_captions = torch.stack([clip_tokenizer(cap)[0] for cap in all_captions], dim=0)
                    tokenized_captions = tokenized_captions.to(device_to_use)
                    embeddings = embed_model.encode_text(tokenized_captions)
                    embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)
                elif args.blip_text_encoder:
                    embeddings = embed_model(image=None, 
                                             caption=all_captions, 
                                             match_head='return_text_features',
                                             device_to_use=device_to_use)
                # separate
                text_embedd = embeddings[0:curr_bt_size, :].view(curr_bt_size, 1, -1)
                generated_embedd = embeddings[curr_bt_size:, :].view(curr_bt_size, num_gen_captions, -1)
                cosine_sim = torch.matmul(text_embedd, generated_embedd.transpose(2,1)).squeeze(dim=1)
                current_scores = torch.amax(cosine_sim, dim=1)   
                
                # get best generated caption
                num_rows, num_cols = cosine_sim.shape
                col_best_score = torch.argmax(cosine_sim, dim=1).cpu().numpy()
                row_best_score = torch.arange(num_rows)
                # map 2d to 1d
                index_1d = ((row_best_score * num_cols) + col_best_score).tolist()
                best_generated_captions = [curr_generated_captions[idx] for idx in index_1d]
                # track results
                best_generated_caption_list.extend(best_generated_captions)
                caption_list.extend(curr_raw_captions)
                uid_list.extend(uids[start_idx:start_idx+batch_size])
                keys_list.extend(keys[start_idx:start_idx+batch_size])
                old_scores.extend(cap_scores[start_idx:start_idx+batch_size])
                # set new_scores for empty generated captions to zero
                current_scores = current_scores.cpu().numpy().tolist()
                for idx, gen_cap in enumerate(best_generated_captions):
                    if len(gen_cap) == 0:
                        current_scores[idx] = 0.0
                        
                new_scores.extend(current_scores)   
    finally:
        queue.put(gpu_id)
    
    
    result = {'uid': uid_list, 
              'key': keys_list,
              'text': caption_list, 
              'generatedtext': best_generated_caption_list,
              'cap_scores': new_scores,
              'old_cap_scores': old_scores}
    
    # write results
    # Convert the dictionary to a Pandas DataFrame
    df = pd.DataFrame.from_dict(result)
    # Convert the DataFrame to a PyArrow Table
    table = pa.Table.from_pandas(df)
    print(output_file_path)
    # Write the Table to Parquet
    pq.write_table(table, output_file_path) 
      
            
def main():
    parser = argparse.ArgumentParser(description='Description of your program.')
    # Add arguments to the parser
    parser.add_argument('--parquet_dir', default="inference_results/test_saveCaptions_medium_10072284_9/",
                        type=str, help='Path to parquet files')
    parser.add_argument('--num_workers', default=10, type=int, help='Number of workers')
    parser.add_argument('--per_device_batch_size', default=256, type=int, help='Global inference batch size')
    parser.add_argument('--sentence_language_model', default='unilingual', type=str, choices=['unilingual', 'multilingual'], 
                        help='choose either a unilingual or a multilingual model for text similarity')
    parser.add_argument("--device", default="cuda", type=str, help="device (Use cuda or cpu Default: cuda)")
    parser.add_argument("--verbose",help="print what each process is seeing", action="store_true")
    parser.add_argument('--output_dir', default="", type=str)
    # select a caption encoder to compute the similarity between a generated caption and an alt-text
    parser.add_argument("--clip_text_encoder", action='store_true', help='Use CLIP text encoder to compute similarity')
    parser.add_argument("--blip_text_encoder", action='store_true', help='Use BLIP text encoder to compute similarity')
    parser.add_argument("--sentence_transformer_text_encoder", action='store_true', help='Use Sentence transformer text encoder to compute similarity')
    

    # Parse the command-line arguments
    args = parser.parse_args()
    print(args)
    # only one text encoder is activated
    assert sum([args.sentence_transformer_text_encoder, args.clip_text_encoder, args.blip_text_encoder]) == 1, (
        "One text encoder must be selected to compute similarity!"
        "This script supports sentence_transformer_text_encoder, clip_text_encoder, blip_text_encoder"
    )
    
    # only assign output_dir if it is not set
    if args.output_dir == "":
        # output folder
        timestamp = datetime.datetime.now().strftime('%H-%M-%S')
        # get an integer from the postfix of the parquet_dir
        reference_num = os.path.basename(os.path.dirname(args.parquet_dir)).split('_')[-1]
        output_dir = os.path.join(args.parquet_dir, f'recomputed_{timestamp}_{reference_num}/')
        assert os.path.exists(output_dir) == False
        os.mkdir(output_dir)
        args.output_dir = output_dir
    
    # multiprocess
    # Create a multiprocessing pool with the number of CPUs available
    num_cpus = multiprocessing.cpu_count()
    num_gpus = torch.cuda.device_count()
    print(f'num GPU: {num_gpus} num CPUs: {num_cpus}')   
    NUM_GPUS = num_gpus
    PROC_PER_GPU = 64 // num_gpus
    print(f'NUM GPU : {NUM_GPUS} PROC_PER_GPU {PROC_PER_GPU} Batch Size {args.per_device_batch_size}')
    parquet_filepaths = glob.glob(f'{args.parquet_dir}/*.parquet')
    inputs = []
    for i in range(0, len(parquet_filepaths)):
        inputs.append((args, parquet_filepaths[i]))

    
    # initialize the queue with the GPU ids
    for gpu_ids in range(NUM_GPUS):
        for _ in range(PROC_PER_GPU):
            queue.put(gpu_ids)
  
    pool = multiprocessing.Pool(NUM_GPUS*PROC_PER_GPU)
    start = time.time()
    for _ in pool.imap_unordered(inference_func, inputs):
        pass
    end = time.time()
    print(end - start)
    pool.close()
    pool.join()
        
    
if __name__ == '__main__':
    main()
            

