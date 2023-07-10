import sys
sys.path.append("../../src")

import argparse
from collections import Counter, namedtuple
import json
from pathlib import Path
from multiprocessing import Pool
from functools import partial

import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer
from torch.utils.data import Dataset, DataLoader

from dataset_retro import RetroDataset, ChunkedSequenceDataset, RetroTrainingExample, ShardedChunkedSequenceDataset, \
    ChunkNeighbourDataset, ShardedChunkNeighbourDataset


RetroTrainingExample = namedtuple("RetroTrainingExample", [
    "input_ids",
    "neighbour_ids", 
    "neighbour_chunk_ids",
    "decoded_neighbours",
])


class ChunkedSequenceDatasetWithIDGetter(ChunkedSequenceDataset):
    def __init__(self, chunks: Path, seq2chunk: Path, chunk2seq: Path, mmap_mode='r'):
        self.chunks_path = chunks
        self.chunks = np.load(str(chunks), mmap_mode=mmap_mode)
        self.seq2chunk = np.load(str(seq2chunk), mmap_mode=mmap_mode)
        self.chunk2seq = np.load(str(chunk2seq), mmap_mode=mmap_mode)

    def get_chunk_ids(self, chunk_index, shard_range_start, include_continuation_chunks=0):
        start_idx = chunk_index
        end_idx = chunk_index + 1
        while end_idx - start_idx - 1 < include_continuation_chunks and \
            end_idx < len(self.chunk2seq) and \
            self.chunk2seq[start_idx] == self.chunk2seq[end_idx]:
            end_idx += 1
        return slice(start_idx+shard_range_start, end_idx+shard_range_start)
        
        
class ShardedChunkedSequenceDatasetWithIDGetter(ShardedChunkedSequenceDataset):  
    def get_chunk_ids(self, chunk_index, include_continuation_chunks: int=0):
        for shard_range, shard in zip(self.shard_chunk_ranges, self.shards):
            if int(chunk_index) in shard_range:
                local_chunk_index = chunk_index - shard_range.start
                return shard.get_chunk_ids(local_chunk_index, shard_range.start, include_continuation_chunks)
        raise IndexError(f"Chunk with index {chunk_index} not found in index")        

        
class ChunkNeighbourDatasetWithIDGetter(ChunkNeighbourDataset):
    def __init__(self, neighbours: Path, retrieval_dataset: ShardedChunkedSequenceDataset, mmap_mode='r'):
        self.neighbours = np.load(str(neighbours), mmap_mode=mmap_mode)
        self.retrieval_dataset = retrieval_dataset

    def get_neighbours_ids(self, chunk_index: int, num_neighbours: int=None, continuation_chunks: int=1):
        return [int(self.retrieval_dataset.get_chunk_ids(neighbour_chunk_idx, continuation_chunks).start) \
                if neighbour_chunk_idx != -1 else None \
                for neighbour_chunk_idx in self.neighbours[chunk_index][:num_neighbours]]

    
class ShardedChunkNeighbourDatasetWithIDGetter(ShardedChunkNeighbourDataset):
    def get_neighbours_ids(self, chunk_index: int, num_neighbours: int=None, continuation_chunks: int=1):
        for shard_range, shard in zip(self.shard_ranges, self.shards):
            if int(chunk_index) in shard_range:
                local_index = chunk_index - shard_range.start
                return shard.get_neighbours_ids(local_index, num_neighbours, continuation_chunks)
        raise IndexError(f"Neighbours for index {chunk_index} not found")    


class RetroIDDataset(RetroDataset):
    def __init__(self, *args, **kwargs):
       self.tokenizer = kwargs.pop('tokenizer') 
       self.analyzer_method = kwargs.pop('analyzer_method')
       self.analyzer = kwargs.pop('analyzer')
       super().__init__(*args, **kwargs)
    
    def __getitem__(self, seq_index: int):
        input_chunk_indices = self.input_dataset.get_chunk_indices_of_sequence(seq_index)
        
        # input_ids
        input_ids = np.concatenate([
            self.input_dataset.get_chunk_tokens(chunk_index)
            for chunk_index in input_chunk_indices[:self.max_num_chunks]
        ])

        # neighbour_ids
        neighbour_ids = [self.neighbour_dataset.get_neighbours(
                            chunk_index, 
                            num_neighbours=None,
                            continuation_chunks=0
                            ) for chunk_index in input_chunk_indices[:self.max_num_chunks]] 

        # neighbour_chunk_ids
        neighbour_chunk_ids = [self.neighbour_dataset.get_neighbours_ids(
                            chunk_index, 
                            num_neighbours=None,
                            continuation_chunks=0
                            ) for chunk_index in input_chunk_indices[:self.max_num_chunks]] 

        # convert to tokens and remove Nones
        no_none_neighbour_chunk_ids = neighbour_chunk_ids 
        no_none_decoded_neighbours = []
        for i, neighbour_ids_batch in enumerate(neighbour_ids):
            no_none_neighbour_ids = [n for n in neighbour_ids_batch if n is not None]
            if len(no_none_neighbour_ids) < len(neighbour_ids_batch):
                no_none_neighbour_chunk_ids[i] = [n for n in neighbour_chunk_ids[i] if n is not None]
            decoded_neighbours_batch = self.tokenizer.batch_decode(no_none_neighbour_ids, skip_special_tokens=True)

            if self.analyzer_method == 't5-base':
                try:
                    encoded_neighbours_batch = self.tokenizer.batch_encode_plus(decoded_neighbours_batch, add_special_tokens=False)
                    decoded_neighbours_batch = [encoded_neighbours_batch.tokens(i) for i in range(len(encoded_neighbours_batch.input_ids))]
                except:
                    decoded_neighbours_batch = [''] 
            elif self.analyzer_method == 'lucene':
                decoded_neighbours_batch = [self.analyzer(n) for n in decoded_neighbours_batch]
            else:
                raise NotImplementedError

            no_none_decoded_neighbours.append(decoded_neighbours_batch)

        return RetroTrainingExample(
            input_ids, 
            neighbour_ids, 
            no_none_neighbour_chunk_ids,
            no_none_decoded_neighbours,
        )


def get_retro_dataset_from_spec(
    spec_file: Path, 
    num_neighbours=None,
    continuation_chunks=1,
    pad_token_idx=0,
    max_len=None,
    tokenizer=None,
    analyzer_method=None,
    analyzer=None,
) -> RetroIDDataset:

    spec = json.load(spec_file.open())
    base_dir = spec_file.parent

    # input dataset
    input_dataset = ShardedChunkedSequenceDatasetWithIDGetter([
        ChunkedSequenceDatasetWithIDGetter(
            chunks=base_dir / shard["chunks"],
            seq2chunk=base_dir / shard["seq2chunk"],
            chunk2seq=base_dir / shard["chunk2seq"]
        )
        for shard in spec["shards"]
    ])

    # retrieval dataset
    index_spec = json.load((base_dir / spec["neighbours"]["index_spec"]).open())
    index_base_dir = base_dir / Path(spec["neighbours"]["index_spec"]).parent
    retrieval_dataset = ShardedChunkedSequenceDatasetWithIDGetter([
        ChunkedSequenceDatasetWithIDGetter(
            chunks=index_base_dir / shard["chunks"],
            seq2chunk=index_base_dir / shard["seq2chunk"],
            chunk2seq=index_base_dir / shard["chunk2seq"]
        )
        for shard in index_spec
    ])

    # neighbour dataset
    neighbour_dataset = ShardedChunkNeighbourDatasetWithIDGetter([
        ChunkNeighbourDatasetWithIDGetter(
            neighbours=base_dir / shard["neighbours"],
            retrieval_dataset=retrieval_dataset
        )
        for shard in spec["shards"]
    ])

    retro_dataset = RetroIDDataset(
        input_dataset=input_dataset,
        neighbour_dataset=neighbour_dataset,
        num_neighbours=num_neighbours,
        continuation_chunks=continuation_chunks,
        pad_token_idx=pad_token_idx,
        max_len=max_len,
        tokenizer=tokenizer,
        analyzer_method=analyzer_method,
        analyzer=analyzer,
    )
    return retro_dataset

 
BM25_scores = [] ###

class BM25Reorder:
    def __init__(self, analyzer_method='t5-base', analyzer=None):
        self.analyzer_method = analyzer_method
        self.analyzer = analyzer
        self.tokenizer = AutoTokenizer.from_pretrained("t5-base", use_fast=True, model_max_length=10000)
        self.df_dict = {term: counts['df'] for term, counts in json.load(open(args.df_dict)).items()}
        #self.df_dict = {term: counts for term, counts in json.load(open(args.df_dict)).items()}

    def _chunker(self, seq: list, chunk_size: int=64):
        for i in range(0, len(seq), chunk_size):
            yield seq[i:i + chunk_size]
    
    def _tfidf_score(self, query: list, neighbour: list, N: int=2766410364) -> float:
        counts = Counter(neighbour)
        intersection = set(neighbour) & set(query)
        tc = np.array([counts[term] for term in intersection])
        df = np.array([self.df_dict[term] for term in intersection])
        score = (tc * np.log(N/df)).sum()
        return score

    def _bm25_score(self, query: list, neighbour: list, N: int=2766410364, avg_d: float=61.18, k1: float=0.9, b: float=0.4) -> float:
        d = len(neighbour)
        counts = Counter(neighbour)
        intersection = set(neighbour) & set(query)
        tc = np.array([counts[term] for term in intersection])
        df = np.array([self.df_dict.get(term, 0) for term in intersection])
        
        idf = np.log(1 + (N - df + 0.5)/(df + 0.5))
        tf = ((k1 + 1) * tc) / (k1 * (1.0 - b + b * (d/avg_d)) + tc)
        score = (tf * idf).sum()
        return score

    def reorder(self, sample, scoring_method='bm25'):
        if scoring_method == 'bm25':
            score_func = self._bm25_score
        elif scoring_method == 'tfidf':
            score_func = self._tfidf_score
        else:
            raise NotImplementedError

        reordered_neighbours = []
        chunk_bm25_scores = [] ###
        for i, chunk in enumerate(self._chunker(sample.input_ids)):
            if self.analyzer_method == 't5-base':
                query = self.tokenizer.encode_plus(self.tokenizer.decode(chunk, skip_special_tokens=True), add_special_tokens=False).tokens()
            elif self.analyzer_method == 'lucene':
                query = self.analyzer(self.tokenizer.decode(chunk, skip_special_tokens=True))

            scores = [score_func(query, neighbour) for neighbour in sample.decoded_neighbours[i]]
            chunk_bm25_scores.append(max(scores)) ###
            reordered_neighbours.append([x for _, x in sorted(zip(scores, sample.neighbour_chunk_ids[i]), \
                                                        key=lambda pair: pair[0], reverse=True)])
        BM25_scores.append(np.mean(chunk_bm25_scores))
        return reordered_neighbours


def main(args):
    if args.analyzer_method == 'lucene':
        from pyserini.analysis._base import Analyzer, get_lucene_analyzer
        analyzer = Analyzer(get_lucene_analyzer()).analyze
    else:
        analyzer = None

    bm25_reorder = BM25Reorder(args.analyzer_method, analyzer)

    val_ds = get_retro_dataset_from_spec(
        spec_file=args.spec_file,
        num_neighbours=0,
        continuation_chunks=0,
        pad_token_idx=0,
        max_len=1024,
        tokenizer=bm25_reorder.tokenizer,
        analyzer_method=args.analyzer_method,
        analyzer=analyzer,
    )

    if args.max_seqs != -1:
        from torch.utils.data import Subset
        from numpy.random import default_rng

        rng = default_rng(0)
        ids = rng.choice(len(val_ds), size=args.max_seqs, replace=False)
        val_ds = Subset(val_ds, ids)

    val_dl = DataLoader(val_ds, num_workers=8, collate_fn=lambda x: x[0])

    reordered_neighbours = []
    for sample in tqdm(val_dl):
        reordered_neighbours.append(bm25_reorder.reorder(sample, scoring_method=args.scoring_method))

    with open('bm25_scores.txt', 'w+') as f:
        f.write('\n'.join([str(s) for s in BM25_scores]))
    exit() ###

    # Reshape and save
    reordered_neighbours = [x + [-1 for _ in range(args.num_neighbours - len(x))] \
                            for y in reordered_neighbours for x in y]
    np.save(args.out, reordered_neighbours)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--spec-file', type=Path, help='Spec JSON file.', required=True)
    parser.add_argument('--index-spec', type=Path, help='Index sepc JSON file.', required=True)
    parser.add_argument('--df-dict', type=Path, help='DF counts JSON file.', required=True)
    parser.add_argument('--out', type=Path, help='.npy file dir for the output.', required=True)
    parser.add_argument('--scoring-method', type=str, default='bm25', help='bm25 or tfidf.')
    parser.add_argument('--analyzer-method', type=str, default='t5-base', help='t5-base or lucene')
    parser.add_argument('--max-seqs', type=int, default=-1, help='-1 for no subsampling.')
    parser.add_argument('--num-neighbours', type=int, default=4)
    args = parser.parse_args()
    main(args)