# SIEVE

*SIEVE:* Multimodal Dataset Pruning Using Image Captioning Models

[[ Paper ]](https://arxiv.org/abs/2310.02110)
<p align="center">
<img src="figs/sieve.jpg" alt="" width="100%"/>
</p>

## Installing dependencies
```
conda env create --name sieve --file=environment.yml
```

To activate the environment:

```
conda activate sieve
```

## Compute alignment scores
The `webdataset_inference.py` script creates a webdataset dataloader and can be used to compute the alignment scores using **SIEVE** and optionally **CLIPScore**. The input shards are distributed amongst available nodes and the output is a `.parquet` file per gpu. The parquet file contains information about each sample and the computed alignment scores. 

1. To compute **SIEVE** scores using `--captioning` option:

```
torchrun --nnodes <NUM_NODES> --nproc_per_node <NUM_GPUS> webdataset_inference.py \
         --data_dir <PATH_TO_SHARD_TAR_FILES> \ 
         --output_dir <OUTPUT_DIR>" \
         --captioning  \
         --model_url "https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model_base_14M.pth" 
```
The `--output_dir` directory will contain the output parquet files. By default we generate 8 captions per image and only the caption  with the higest sentence similarity is saved in the parquet files. To save all generated captions add option `--save_all_captions`. This is usefull for post processing (i.e., removing medium words and then recomputing alignment scores or recomputed alignment scores using different sentence encoders). 

2. To compute **SIEVE** and **CLIPScore** alignment scores use `--—clipcap` option:
```
torchrun --nnodes <NUM_NODES> --nproc_per_node <NUM_GPUS> webdataset_inference.py \
         --data_dir <PATH_TO_SHARD_TAR_FILES> \ 
         --output_dir <OUTPUT_DIR>" \
         —-clipcap \
         --model_url "https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model_base_14M.pth"
```
Finally, you can also use `--cliping` to only compute **CLIPScore**.

## Compute SIEVE score using Sentence Transformer, CLIP Text Encoder or BLIP Text Encoder (optional)
The `sentence_similarity_inference.py` script can be used to re-compute SIEVE scores using Sentence Transformer, CLIP Text Encoder or BLIP Text Encoder. The script processes a directory of parquet files generated by the `webdataset_inference.py` script. To use the `sentence_similarity_inference.py` script, `--save_all_captions` must be passed to `webdataset_inference.py` to save all generated captions so that we can re-compute the alignment scores using different encoders. This scripts also enables masking medium phrases. 

1. To compute alignment score in the embedding space of **Sentence Transformer**:
```
python sentence_similarity_inference.py \
       --sentence_transformer_text_encoder \
       --parquet_dir <PATH_TO_PARQUET_FILES>
```
Here, set `--parquet_dir` to the directory containing parquet files generated by `webdataset_inference.py`

2. To compute alignment score in the embedding space of **CLIP** text encoder:
```
python sentence_similarity_inference.py \
    --clip_text_encoder \
    --parquet_dir <PATH_TO_PARQUET_FILES>
```
3. To compute alignment score in the embedding space of **BLIP** text encoder:
```
python sentence_similarity_inference.py \
       --blip_text_encoder \
       --parquet_dir <PATH_TO_PARQUET_FILES>
```

## Create Datasets using SIEVE, CLIPScore and SIEVE+CLIPScore

1. To generate a dataset that keeps the top 20% of the image-text pairs based on **SIEVE** scores:

```
python baselines.py --name generic_filter  \
                    --metadata_dir <PATH_TO_PARQUET_FILES>\
                    --keys_to_fuse cap_scores \
                    --weights_per_key 1.0 \
                    --fraction 0.2 \
                    --save_path <NAME_OF_DATASET.npy> 
```
Using `--keys_to_fuse` you can choose which columns in the parquet files to use for pruning. Here, *cap_scores* is the column associated with **SIEVE** scores in the parquet files. 
This script will save two files; 1) `.npy` file containing the unique sample ids, 2) `.pkl` file saves keys associated with unique ids. This enables completely skipping the resharding of the data everytime you create a new dataset. We use the `.pkl` in openclip dataloader to skip loading any samples not in the `.pkl` file. 

2. To generate a dataset that keeps the top 20% of the image-text pairs based on **CLIPScore** scores:
```
python baselines.py --name generic_filter  \
                    --metadata_dir <PATH_TO_PARQUET_FILES>\
                    --keys_to_fuse clip_scores \
                    --weights_per_key 1.0 \
                    --fraction 0.2 \
                    --save_path <NAME_OF_DATASET.npy> 
```
3. To generate a dataset that keeps the top 20% of the image-text pairs based fusing **SIEVE** and **CLIPScore** as described in the paper:
```
python baselines.py --name generic_filter  \
                    --metadata_dir \
                    --keys_to_fuse cap_scores clip_scores \
                    --weights_per_key 0.5 0.5 \
                    --fraction 0.2 \
                    --save_path <NAME_OF_DATASET.npy>
```
Here, each pruning signal is normalized independently and then fused with a weight of 0.5 on each signal. More generally, the *generic_filter* enables fusing an arbitrary number of pruning signals with weights defined by `--weights_per_key`


## Train using TRIE files (No resharding required)
To prevent resharding the data using the generated filters. We have adapted openclip dataloader to use a dictionary passed to the training script using the *subset_file* argument to determine whether a sample should be included in training. 

To pretrain clip model on the *medium* scale of datacomp using the generated `.pkl` using multiple node run:

```
sbatch  scripts/slurm_train.sh \
        --data_dir <PATH_TO_SHARD_TAR_FILES> \
        --subset_file <PATH_TO_PICKLE_FILE> \
        --scale medium \
        --seed 1
```
Here, *subset_file* is an argument that can be used to pass the generated `.pkl` containing the *keys* of the selected samples. 

## Evaluating pretrained CLIP models

To evaluate pretrained CLIP models on 38 downstream tasks:

```
sbatch scripts/slurm_eval.sh \
        -f <PATH_TO_PRETRAINED_FOLDER>
```


The majority of SIEVE is licensed under CC-BY-NC, however portions of the project are available under separate license terms: datacomp is licensed under the MIT license; open_clip is licensed under https://github.com/mlfoundations/open_clip/blob/main/LICENSE; BLIP is licensed under the BSD 3-clause license.


## Citation

If you use SIEVE, please use the below BibTex for citing:
:
```text
@article{mahmoud2023sieve,
  title={SIEVE: Multimodal Dataset Pruning Using Image Captioning Models},
  author={Mahmoud, Anas and Elhoushi, Mostafa and Abbas, Amro and Yang, Yu and Ardalani, Newsha and Leather, Hugh and Morcos, Ari},
  journal={arXiv preprint arXiv:2310.02110},
  year={2023}
}
```