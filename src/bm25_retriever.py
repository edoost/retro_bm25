import sys
sys.path.append("../retro/src")

import argparse
from collections import defaultdict
import json
from pathlib import Path
import time
from multiprocessing import Pool

import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer
from pyserini.search.lucene import LuceneSearcher

from train_retro import get_retro_dataset_from_spec
from dataset_retro import ShardedChunkedSequenceDataset


def chunker(seq: list, chunk_size: int=64):
    for i in range(0, len(seq), chunk_size):
        yield seq[i:i + chunk_size]


def get_global_chunk2seq(index_spec: Path):
    shards = [np.load(index_spec.parent / shard['chunk2seq']) \
              for shard in json.load(index_spec.open())]
    global_chunk2seq = np.zeros(sum([shard.shape[0] for shard in shards]), dtype=int)
    shard_offset, current_idx = 0, 0
    for shard in shards:
        global_chunk2seq[current_idx:current_idx+shard.shape[0]] = shard + shard_offset
        shard_offset += shard[-1] + 1
        current_idx += shard.shape[0]
    return global_chunk2seq


def get_val_global_seq_ids(index_spec: Path, val_spec: Path):
    index_shards_names = [shard['chunk2seq'].split('/')[-1] for shard in json.load(index_spec.open())]
    index_shards = [np.load(index_spec.parent / shard['chunk2seq'], mmap_mode='r') \
                    for shard in json.load(index_spec.open())]
    with Pool() as pool:
        index_shards_unique = pool.map(np.unique, index_shards[:-1])
    index_shards_start_ids = np.cumsum([0] + [shard.shape[0] for shard in index_shards_unique])

    val_shards_names = [shard['chunk2seq'].split('/')[-1] for shard in json.load(val_spec.open())['shards']]
    val_shards = [np.load(val_spec.parent / shard['chunk2seq'], mmap_mode='r') \
                  for shard in json.load(val_spec.open())['shards']]
    with Pool() as pool:
        val_shards_unique = pool.map(np.unique, val_shards)

    val_global_seq_ids = np.zeros(sum(shard.shape[0] for shard in val_shards_unique), dtype=int)
    current_idx = 0
    for i in range(len(val_shards_unique)):
        offset = index_shards_start_ids[index_shards_names.index(val_shards_names[i])]
        val_global_seq_ids[current_idx:current_idx+val_shards_unique[i].shape[0]] = val_shards_unique[i] + offset
        current_idx += val_shards_unique[i].shape[0]

    return val_global_seq_ids


def main(args):
    val_ds = get_retro_dataset_from_spec(
        spec_file=args.val_spec,
        num_neighbours=0,
        continuation_chunks=0,
        pad_token_idx=0,
        max_len=1024
    )

    from torch.utils.data import Subset
    from numpy.random import default_rng

    rng = default_rng(0)
    ids = rng.choice(len(val_ds), size=args.max_seqs, replace=False)
    val_ds = Subset(val_ds, ids)

    global_chunk2seq = get_global_chunk2seq(args.index_spec)
    val_global_seq_ids = get_val_global_seq_ids(args.index_spec, args.val_spec)

    tokenizer = AutoTokenizer.from_pretrained("t5-base", use_fast=True, model_max_length=10000)
    searchers = (LuceneSearcher(str(index_path)) for index_path in args.index_path.iterdir())

    chunks = []
    qid2id = {}
    for i, seq in enumerate(val_ds):
        chunked_seq = chunker(seq.input_ids) 
        for chunk in chunked_seq:
            qid2id[len(chunks)] = ids[i]
            chunks.append(chunk)

    text_chunks = tokenizer.batch_decode(chunks, skip_special_tokens=True)

    print("Searching ...")
    b = time.time()
    qids = [str(qid) for qid in range(len(text_chunks))]
    results = [searcher.batch_search(text_chunks, qids=qids, k=args.topk, threads=args.threads) \
                for searcher in searchers]
    #results = [searcher.search(text_chunks[0], k=args.topk) for searcher in searchers] 
    print('time:', time.time() - b)

    retrieved = []
    for qid in qids: 
        hits = [(int(hit.docid), hit.score) for hits in results for hit in hits[qid]]
        filtered_hits = [hit for hit in hits if global_chunk2seq[hit[0]] != val_global_seq_ids[qid2id[int(qid)]]]
        top_hits = sorted(filtered_hits, key=lambda x: x[1], reverse=True)[:args.topk]
        top_hits = [top_hit[0] for top_hit in top_hits]
        top_hits.extend([-1 for _ in range(args.topk - len(top_hits))])
        retrieved.append(top_hits)
    
    np.save(args.out_dir, np.asarray(retrieved))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--index-path', type=Path, help='Lucene index path.', required=True)
    parser.add_argument('--val-spec', type=Path, help='Validation sepc JSON file.')
    parser.add_argument('--index-spec', type=Path, help='Index sepc JSON file.')
    parser.add_argument('--out-dir', type=Path, help='Output directory (.npy).')
    parser.add_argument('--topk', type=int, default=4)
    parser.add_argument('--threads', type=int, default=256)
    parser.add_argument('--max-seqs', type=int, default=1000)
    args = parser.parse_args()
    main(args)