# Copyright (c) Meta Platforms, Inc. and affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import os
from functools import partial
import multiprocessing as mp
from typing import Any, Set, Tuple, List, Union

import fasttext
import fsspec
import gcld3
import nltk
import numpy as np
import pandas as pd
import torch
from nltk.corpus import wordnet
from tqdm import tqdm
import time
from queue import Empty
import requests
from PIL import Image
import glob as glob

from baselines.utils import download, worker_threadpool, worker_threadpool_only_uid_int
from baselines.build_trie_file import build_and_save_trie

fasttext.FastText.eprint = lambda x: None
import pickle


def get_fasttext_language(text: str, lang_detect_model: Any) -> str:
    """helper to detect language of a piece of text (fasttext)

    Args:
        text (str): text whose language we want to determing
        lang_detect_model (Any): fasttext model to detect langauge

    Returns:
        str: ISO language code
    """
    text = text.replace("\n", " ")
    language = lang_detect_model.predict(text)[0][0].split("__label__")[1]

    return language


def get_gcld3_language(text: str, gcld3_model) -> str:
    """helper to detect language of a piece of text (gcld3).
    Note: this is only used for our LAION-2B filtering reproduction

    Args:
        text (str): text whose language we want to determing
        lang_detect_model (Any): fasttext model to detect langauge

    Returns:
        str: ISO language code
    """
    text = text.replace("\n", " ")
    language = gcld3_model.FindLanguage(text=text).language

    return language


def caption_filter(df: pd.DataFrame, lang_detect_model: Any) -> np.ndarray:
    """apply a low-level text filter for the image based baseline

    Args:
        df (pd.DataFrame): parquet metadata
        lang_detect_model (Any): fasttext model

    Returns:
        np.ndarray: boolean numpy array containing selected entries
    """
    caption_num_words = df.text.apply(lambda x: len(fasttext.tokenize(x)))
    caption_num_chars = df.text.apply(len)

    lang_preds, _ = lang_detect_model.predict(
        [x.replace("\n", " ") for x in df.text.values], k=1
    )
    fasttext_en = [x[0].replace("__label__", "") == "en" for x in lang_preds]

    mask = fasttext_en & (caption_num_words > 1) & (caption_num_chars > 5)

    return mask.to_numpy()


@torch.no_grad()
def get_centroid_ids_gpu(
    features: torch.Tensor, centroids: torch.Tensor, batch_size: int, device: int
) -> torch.Tensor:
    """assign features to closest centroid

    Args:
        features (torch.Tensor): features to assign to centroids
        centroids (torch.Tensor): reference centroids
        batch_size (int): gpu batch size
        device (int): gpu number

    Returns:
        torch.Tensor: assignment of features to labels
    """
    device_string = f"cuda:{device}"
    centroids_gpu = centroids.to(device_string)
    labels = torch.zeros(features.shape[0], dtype=torch.long)

    for i in range(0, features.shape[0], batch_size):
        similarity = torch.einsum(
            "ik, jk -> ij",
            features[i : i + batch_size, :].float().to(device_string),
            centroids_gpu,
        )
        matches = torch.argmax(similarity, dim=1).cpu()
        labels[i : i + batch_size] = matches.long()

    return labels


def image_filter_helper(
    pool_centroids: torch.Tensor,
    target_centroid_ids: torch.Tensor,
    batch_size: int,
    device_index: int,
    in_queue: mp.Queue,
    out_queue: mp.Queue,
    arch: Union[str, None] = None,
    threshold: Union[float, None] = None,
) -> None:
    """worker function to image_based filtering, pulling off a queue of tasks

    Args:
        pool_centroids (torch.Tensor): centroids derived from k-means on a pool (e.g., the small pool)
        target_centroid_ids (torch.Tensor): target centroid indices of interest, only want samples nearest to these centroids
        batch_size (int): gpu batch size for assigning samples loaded from the in_queue to pool centroids
        device_index (int): device on which to run the gpu processing
        in_queue (mp.Queue): task queue with fsspec, metadata path pairs
        out_queue (mp.Queue): output queue to send filtred uids
        arch: (Union[str, None]): If specified, we want to apply a threshold to arch=B/32 or L/14 clip scores. Defaults to None.
        threshold: (Union[float, None]): threshold to apply over arch clip scores. Defaults to None.
    """
    while True:
        fs_root = None
        try:
            fs_root = in_queue.get(timeout=1)
        except Empty:
            # case where the queue is depleated, worker should return
            break

        fs, path_root = fs_root
        lang_detect_model = fasttext.load_model(
            download("fasttext", "~/.cache/fasttext")
        )

        df = None
        df_index = None

        if arch is not None:
            key = "clip_l14_similarity_score"
            if arch == "b32":
                key = "clip_b32_similarity_score"

            df = pd.read_parquet(
                f"{path_root}.parquet", columns=["uid", "text", key], filesystem=fs
            )
            df_index = df[key] >= threshold
            df = df[df_index]
        else:
            df = pd.read_parquet(
                f"{path_root}.parquet", columns=["uid", "text"], filesystem=fs
            )

        candidate_embedding = None
        with fs.open(f"{path_root}.npz") as f:
            candidate_embedding = torch.from_numpy(np.load(f)["l14_img"])

            if df_index is not None:
                candidate_embedding = candidate_embedding[df_index]

        # simple caption filter first
        mask = caption_filter(df, lang_detect_model)

        uids = df.uid[mask]

        candidate_centroid_ids = get_centroid_ids_gpu(
            candidate_embedding[mask],
            pool_centroids,
            batch_size,
            device_index,
        )

        centroid_id_to_uids = {}
        for uid, label in zip(uids, candidate_centroid_ids):
            centroid_id_to_uids.setdefault(label.item(), []).append(uid)

        uids_to_keep = []
        for i in target_centroid_ids:
            if i.item() in centroid_id_to_uids:
                uids_to_keep.extend(centroid_id_to_uids[i.item()])

        out_queue.put(
            np.array(
                [(int(uid[:16], 16), int(uid[16:32], 16)) for uid in uids_to_keep],
                np.dtype("u8,u8"),
            )
        )


def load_uids_with_basic_filter_helper(fs_url: Tuple[Any, str]) -> np.ndarray:
    """helper to run basic filter on a single parquet

    Args:
        fs_url (Tuple[Any, str]): pair of fsspec file system and parquet url

    Returns:
        np.ndarray: array of uids
    """
    fs, url = fs_url
    df = pd.read_parquet(
        url, columns=["uid", "key", "text", "original_width", "original_height"], filesystem=fs
    )

    lang_detect_model = fasttext.load_model(download("fasttext", "~/.cache/fasttext"))

    fasttext_lang_pred = df.text.apply(
        lambda x: get_fasttext_language(x, lang_detect_model)
    )
    caption_num_words = df.text.apply(lambda x: len(x.split()))
    caption_num_chars = df.text.apply(lambda x: len(x))
    uid_int = df.uid.apply(int, base=16)
    uid_upper_uint64 = (uid_int // 2**64).astype("uint64")
    uid_lower_uint64 = (uid_int % 2**64).astype("uint64")

    inds_array = np.array(list(zip(uid_upper_uint64, uid_lower_uint64)), "u8,u8")

    english_mask = fasttext_lang_pred == "en"
    caption_mask = (caption_num_words > 2) & (caption_num_chars > 5)
    min_image_dim = np.minimum(df.original_width, df.original_height)
    max_image_dim = np.maximum(df.original_width, df.original_height)
    aspect_ratio = max_image_dim / min_image_dim
    image_mask = (min_image_dim >= 200) & (aspect_ratio <= 3.0)

    keys = df['key'][english_mask & caption_mask & image_mask]
    int_uid = inds_array[english_mask & caption_mask & image_mask]
    
    return int_uid, keys

def does_contain_text_entity(text: str, entity_set: Set) -> bool:
    """helper to check if words in text are contained in an entity set

    Args:
        text (str): caption from an image-text pair
        entity_set (Set): set of synset keys we are cross referencing against

    Returns:
        bool: True if any word of text is in the entity set else False
    """
    word_list = text.split()

    for word in word_list:
        synsets = wordnet.synsets(word)
        if len(synsets) == 0:
            continue

        # retrieve the most likely lemma representing the synset
        synset = synsets[0]
        synset_key = synset.offset()
        if synset_key in entity_set:
            return True

    return False


def load_uids_with_text_entity_helper(
    fs_url: Tuple[Any, str], entity_set: Set
) -> np.ndarray:
    """helper for text based filter on a single parquet

    Args:
        fs_url (str): pair of fsspec file system and parquet url
        entity_set (Set): set of synset keys we are referencing against

    Returns:
        np.ndarray: array of uids
    """
    fs, url = fs_url
    lang_detect_model = fasttext.load_model(download("fasttext", "~/.cache/fasttext"))

    df = pd.read_parquet(url, columns=["uid", "key", "text"], filesystem=fs)
    fasttext_lang_pred = df.text.apply(
        lambda x: get_fasttext_language(x, lang_detect_model)
    )
    contains_in21k_synset = df.text.apply(
        lambda x: does_contain_text_entity(x, entity_set)
    )

    uid_int = df.uid.apply(int, base=16)
    uid_upper_uint64 = (uid_int // 2**64).astype("uint64")
    uid_lower_uint64 = (uid_int % 2**64).astype("uint64")

    inds_array = np.array(list(zip(uid_upper_uint64, uid_lower_uint64)), "u8,u8")

    english_mask = fasttext_lang_pred == "en"
    in21k_mask = contains_in21k_synset == True
    
    keys = df['key'][english_mask & in21k_mask]
    int_uid = inds_array[english_mask & in21k_mask]

    return int_uid, keys


def load_uids_helper(fs_url: Tuple[Any, str]) -> np.ndarray:
    """helper to read a parquet and load the uids

    Args:
        fs_url (Tuple[Any, str]): pair of fsspec file system and parquet url

    Returns:
        np.ndarray: array of uids
    """
    fs, url = fs_url
    df = pd.read_parquet(url, columns=["uid", "key"], filesystem=fs)

    keys = df["key"]
    int_uid = np.array(
        [(int(uid[:16], 16), int(uid[16:32], 16)) for uid in df["uid"].values],
        np.dtype("u8,u8"),
    )
    
    return int_uid, keys


def load_metadata(
    metadata_dir_path: str, num_workers: int, columns: List[str] = None
) -> pd.DataFrame:
    """load metadata for many parquets

    Args:
        metadata_dir_path (str): directory where metadata is stored
        num_workers (int): number of cpu workers, each of which processes a parquet
        columns (List[str], optional): list of columns to retain from the parquet. Defaults to None.

    Returns:
        pd.DataFrame: loaded parquet columns
    """
    fs, url = fsspec.core.url_to_fs(metadata_dir_path)
    parquet_paths = [str(x) for x in fs.ls(url) if ".parquet" in x]
    
    worker = partial(pd.read_parquet, columns=columns, filesystem=fs)

    return worker_threadpool_only_uid_int(worker, pd.concat, parquet_paths, num_workers)


def load_uids(metadata_dir_path: str, num_workers: int) -> np.ndarray:
    """load all uids in a metadata containing directory

    Args:
        metadata_dir_path (str): directory where metadata is stored
        num_workers (int): number of cpu workers, each of which processes a parquet

    Returns:
        np.ndarray: array of uids
    """
    fs, url = fsspec.core.url_to_fs(metadata_dir_path)
    parquet_paths = [(fs, str(x)) for x in fs.ls(url) if ".parquet" in x]

    return worker_threadpool(
        load_uids_helper, np.concatenate, parquet_paths, num_workers
    )


def load_uids_with_basic_filter(metadata_dir_path: str, num_workers: int) -> np.ndarray:
    """basic filter from the datacomp paper

    Args:
        metadata_dir_path (str): directory where metadata is stored
        num_workers (int): number of cpu workers, each of which processes a parquet

    Returns:
        np.ndarray: array of uids
    """
    fs, url = fsspec.core.url_to_fs(metadata_dir_path)
    parquet_paths = [(fs, str(x)) for x in fs.ls(url) if ".parquet" in x]

    # download fasttext so that all workers dont't try to download at once
    download("fasttext", "~/.cache/fasttext")

    return worker_threadpool(
        load_uids_with_basic_filter_helper, np.concatenate, parquet_paths, num_workers
    )


def load_uids_with_text_entity(metadata_dir_path: str, num_workers: int) -> np.ndarray:
    """text based filter from the datacomp paper

    Args:
        metadata_dir_path (str): directory where metadata is stored
        num_workers (int): number of cpu workers, each of which processes a parquet

    Returns:
        np.ndarray: array of uids
    """
    entity_ids = open(download("imagenet21k_wordnet_ids"), "r").readlines()
    entity_ids = [x.strip() for x in entity_ids]
    entity_ids = [int(x[1:]) for x in entity_ids]

    fs, url = fsspec.core.url_to_fs(metadata_dir_path)
    parquet_paths = [(fs, str(x)) for x in fs.ls(url) if ".parquet" in x]

    # download fasttext so that all workers dont't try to download at once
    download("fasttext", "~/.cache/fasttext")

    worker = partial(load_uids_with_text_entity_helper, entity_set=entity_ids)

    return worker_threadpool(worker, np.concatenate, parquet_paths, num_workers)

# min/max normalized
def normalize_df(df, df_min, df_max):
    return (df-df_min)/(df_max-df_min)  

# this function is a generic function that fuses pruning signals 
def load_uids_generic_filter(metadata_dir_path: str, fraction: float, keys_to_fuse: list, weights_per_key: list, num_workers: int):
    num_keys_to_fuse = len(keys_to_fuse)
    assert num_keys_to_fuse == len(weights_per_key)
    assert sum(weights_per_key) == 1.0
    # load parquet for entire data to get threshold needed to achieve top fraction of the data
    df = load_metadata(metadata_dir_path, num_workers=num_workers, columns=keys_to_fuse)
    prune_signals_min = []
    prune_signals_max = []
    for idx in range(0, num_keys_to_fuse):
        key = keys_to_fuse[idx]
        prune_signals_min.append(df[key].min())
        prune_signals_max.append(df[key].max())
        # weighted average of columns
        if idx == 0:
            df['weighted_avg'] = (weights_per_key[idx] * normalize_df(df[key], df_min=df[key].min(), df_max=df[key].max()))
        else:
            df['weighted_avg'] += (weights_per_key[idx] * normalize_df(df[key], df_min=df[key].min(), df_max=df[key].max()))
    
    # find weight that achieves fraction requested
    sorted_df = -np.sort(-df['weighted_avg'].values)
    n = int(len(df['weighted_avg']) * fraction)
    weighted_threshold = sorted_df[n]
    print(f'Keys to fuse: {keys_to_fuse}')
    print(f'Weights: {weights_per_key}')
    print(f'Pruning signal minimum: {prune_signals_min}')
    print(f'Pruning signal maximum: {prune_signals_max}')
    print(f'Threshold for inclusion: {weighted_threshold}')
    worker = partial(load_uids_generic_filter_helper, keys_to_fuse=keys_to_fuse, weights_per_key=weights_per_key, 
                                                prune_signals_min=prune_signals_min, 
                                                prune_signals_max=prune_signals_max, 
                                                weighted_threshold=weighted_threshold)
    fs, url = fsspec.core.url_to_fs(metadata_dir_path)
    parquet_paths = [(fs, str(x)) for x in fs.ls(url) if ".parquet" in x]
    return worker_threadpool(worker, np.concatenate, parquet_paths, num_workers)
     
    
    
def load_uids_generic_filter_helper(fs_url: Tuple[Any, str], keys_to_fuse: list, weights_per_key: list, 
                                 prune_signals_min: list, prune_signals_max: list, weighted_threshold: float):
    
    fs, url = fs_url
    df = None
    df = pd.read_parquet(url, columns=["uid", "key"]+keys_to_fuse, filesystem=fs)
    # normalize each column based on min and max
    assert len(keys_to_fuse) == len(prune_signals_min)
    num_keys_to_fuse = len(keys_to_fuse)
    for idx in range(0, num_keys_to_fuse):
        key = keys_to_fuse[idx]
        df[key] = weights_per_key[idx] * normalize_df(df[key], df_min=prune_signals_min[idx], df_max=prune_signals_max[idx])
        if idx == 0:
            df['weighted_avg'] = df[key]
        else:
            df['weighted_avg'] += df[key]
            
    uid_int = df.uid.apply(int, base=16)
    uid_upper_uint64 = (uid_int // 2**64).astype("uint64")
    uid_lower_uint64 = (uid_int % 2**64).astype("uint64")

    inds_array = np.array(list(zip(uid_upper_uint64, uid_lower_uint64)), "u8,u8")

    mask_low_values = df['weighted_avg'] >= weighted_threshold
    
    keys = df['key'][mask_low_values]
    int_uid = inds_array[mask_low_values]
            
    return int_uid, keys


def apply_filter(args: Any) -> None:
    """function to route the args to the proper baseline function

    Args:
        args (Any): commandline args

    Raises:
        ValueError: unsupported name
    """
    mp.set_start_method("spawn", force=True)

    uids = None
    print(f"running: {args.name}")

    if args.name == "no_filter":
        int_uids, keys = load_uids(
            args.metadata_dir,
            args.num_workers,
        )
    elif args.name == "basic_filter":
        int_uids, keys = load_uids_with_basic_filter(
            args.metadata_dir,
            args.num_workers,
        )
    elif args.name == "text_based":
        nltk.download("wordnet")
        int_uids, keys = load_uids_with_text_entity(
            args.metadata_dir,
            args.num_workers,
        )
    elif args.name == "generic_filter":
        int_uids, keys = load_uids_generic_filter(
            metadata_dir_path=args.metadata_dir,
            fraction=args.fraction[0],
            keys_to_fuse=args.keys_to_fuse,
            weights_per_key=args.weights_per_key,
            num_workers=args.num_workers,
        )      
    else:
        raise ValueError(f"Unknown args.name argument: {args.name}")

    
    number_of_samples = len(int_uids)
    print(f"sorting {len(int_uids)} uids")
    indices_to_sort = int_uids.argsort() # can be used to order keys
    int_uids = int_uids[indices_to_sort] # sorted uids
    # uids
    npy_path = args.save_path[0:-4]+ f'_N{number_of_samples/1e6:.2f}'+'.npy'
    print(f"saving {npy_path} with {len(int_uids)} entries")
    np.save(npy_path, int_uids)
    
    
    # make sure keys are unique
    if keys != None:
        number_of_keys = len(keys)
        assert len(set(keys)) == number_of_samples
        assert number_of_keys == number_of_samples    
        # save
        trie_file_path = args.save_path[0:-4] + f'_N{number_of_samples/1e6:.2f}' + '_trie.pkl'
        print(f"saving {trie_file_path} with {number_of_samples} entries")
        build_and_save_trie(data=keys, file_path=trie_file_path)
        
        if args.save_sorted_keys:
            sorted_keys = [keys[i] for i in indices_to_sort]
            # save list of sorted keys
            print('Saving list of sorted keys ...')
            file_path = args.save_path[0:-4] + f'_N{number_of_samples/1e6:.2f}' + '_sorted_keys.pkl'
            with open(file_path, 'wb') as f:
                pickle.dump(sorted_keys, f)
            print("Saved")
            
            
            

    
    


        
