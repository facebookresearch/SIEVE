# Copyright (c) Meta Platforms, Inc. and affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import os
from pathlib import Path

import torch

from baselines.apply_filter import apply_filter

BASELINES = {
    "no_filter",
    "basic_filter",
    "text_based",
    "image_based",
    "generic_filter"
}

CLUSTER_CENTROID_SCALES = [
    "small",
    "medium",
    "large",
    "xlarge",
]

def check_args(args):
    if args.name not in BASELINES:
        raise ValueError(f"--name must be in: {BASELINES}")

    # image_based checks
    if args.image_based_scale is None and "image_based" in args.name:
        raise ValueError(
            "--image_based_scale value must be passed for image_based and image_based_intersect_clip_score_* baselines (for clustering)"
        )
    if args.image_based_scale is not None and not ("image_based" in args.name):
        raise ValueError(
            "--image_based_scale should only be passed for image_based and image_based_intersect_clip_score_* baselines (for clustering)"
        )
    if "image_based" in args.name and not torch.cuda.is_available():
        raise ValueError(
            "gpus needed for image_based baselines, torch.cuda.is_available() must return true"
        )

    # generic filter checks
    if "generic_filter" in args.name:
        assert args.keys_to_fuse is not None
        assert args.weights_per_key is not None
        assert args.fraction is not None and len(args.fraction) == 1
        # Number of keys to fuse must match number of weights
        assert len(args.keys_to_fuse) == len(args.weights_per_key)
        assert sum(args.weights_per_key) == 1.0

    npy_parent = Path(args.save_path).parent
    if not os.path.exists(npy_parent):
        print(f"creating: {npy_parent}")
        os.mkdir(npy_parent)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="This is a command line script for reproducing the main DataComp filtering baselines. The output of the script is a numpy file (.npy) containing the uids in the filtered subsets in sorted binary format. Please see README.md for additional information"
    )

    parser.add_argument(
        "--name",
        type=str,
        required=True,
        choices=list(BASELINES),
        help="name of the baseline",
    )

    parser.add_argument(
        "--metadata_dir",
        type=str,
        required=True,
        help="directory (local or cloud) containing parquet, npz metadata",
    )

    parser.add_argument(
        "--save_path",
        type=str,
        required=True,
        help="path to output .npy, note: cloudpaths are not supported for this arg",
    )

    parser.add_argument(
        "--num_workers",
        type=int,
        required=False,
        default=os.cpu_count(),
        help="number of workers, generally set to number of cpu cores. workers read their metadata files and filter them in parallel).",
    )

    parser.add_argument(
        "--num_gpus",
        type=int,
        required=False,
        default=torch.cuda.device_count(),
        help="number of gpus for the image_based gpu implementation. num_gpus metadata files are processed in parallel on each gpu worker. NOTE: this parameter is ignored for non-image_basesd baselines",
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        required=False,
        default=1024,
        help="batch size for the image_based gpu implementation. NOTE: this parameter is ignored for non-image_basesd baselines",
    )

    parser.add_argument(
        "--image_based_scale",
        type=str,
        required=False,
        choices=CLUSTER_CENTROID_SCALES,
        help="datacomp scale, used for the clutering baselines",
        default=None,
    )
    # 
    parser.add_argument(
        "--keys_to_fuse",
        type=str, 
        nargs='+', 
        required=False,
        help="a list of  name of columns in parquet files to be fused, \
        e.g. of columns: clip_scores, cap_scores" 
    )
    parser.add_argument(
        "--weights_per_key",
        type=float, 
        nargs='+', 
        required=False,
        help="a list of weights for each column to be fused in keys_to_fuse. This should sum up to 1.0" 
    )
    
    parser.add_argument(
        "--fraction", 
        type=float, 
        nargs='+', 
        help='top percent of samples to keep according to score',
        )
    
    parser.add_argument(
        "--save_sorted_keys", 
        action="store_true", 
        help="save a copy of per sample key in the same order as uids",)
    

    args = parser.parse_args()

    # all error checking happens here and apply_filter assumes correct input
    check_args(args)

    # route the args to the correct baseline
    apply_filter(args)
