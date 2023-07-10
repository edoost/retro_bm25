import argparse
import json
from pathlib import Path

from tqdm import tqdm
import numpy as np
from transformers import AutoTokenizer
from torch.utils.data import DataLoader


def main(args):
    chunk_tokenizer = AutoTokenizer.from_pretrained("t5-base", use_fast=True, model_max_length=10000)

    spec = json.load(args.spec.open("r"))
    base_dir = args.spec.parent
    chunks_paths = [np.load(base_dir / shard['chunks'], mmap_mode="r") for shard in spec]
    total = sum([shard.shape[0] for shard in chunks_paths])

    # Just to pre-tokenize upcoming shards        
    data_loader = DataLoader(chunks_paths, 
                                num_workers=3,
                                collate_fn=lambda x: chunk_tokenizer.batch_decode(x[0], skip_special_tokens=True),
                                prefetch_factor=1,
                                )

    args.output.mkdir()
    doc_id = 0
    # Adding shards
    with tqdm(total=total) as pbar:
        for i, chunks in enumerate(data_loader):
            o_dir = args.output / Path(spec[i]['chunks'].split('/')[-1].split('.')[0] + '.jsonl')
            with open(o_dir, 'w+') as out_file:
                for decoded_chunk in chunks:
                    dic = {'id': str(doc_id), 'contents': decoded_chunk}
                    json.dump(dic, out_file)
                    out_file.write('\n')
                    doc_id += 1
                    pbar.update(1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--spec', type=Path, help='Input spec.')
    parser.add_argument('--soft-split', type=int, help='Split limit.')
    parser.add_argument('--output', type=Path, help='Output path.')
    args = parser.parse_args()
    main(args)