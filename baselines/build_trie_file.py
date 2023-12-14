# Copyright (c) Meta Platforms, Inc. and affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import pickle
from tqdm import tqdm


def dicttrie(arr):
    trie = {}
    for p in tqdm(arr):
        key = p.strip()
        trie[key] = True
    return trie

def trie_search(trie, uid):
    try:
        if trie[uid]:
            return True
    except KeyError:
        return False
    
def build_and_save_trie(data: list, file_path: str):
    print('Building trie structue for the pruned data')
    mytrie = dicttrie(data)
    print('Saving trie.pickle ...')
    with open(file_path, 'wb') as f:
        pickle.dump(mytrie, f)
    print("Saved")
