import argparse
from pathlib import Path

import numpy as np


def main(args):
    if args.lt_over_lb is None:
        from train_retro import get_retro_dataset_from_spec
        from transformers import AutoTokenizer
        val_ds = get_retro_dataset_from_spec(
                    spec_file=args.spec_file,
                    num_neighbours=0,
                    continuation_chunks=0,
                    pad_token_idx=0,
                    max_len=1024
                    )

        tokenizer = AutoTokenizer.from_pretrained("t5-base", use_fast=True, model_max_length=10000)
        
        ids = [seq.input_ids for seq in val_ds]

        #decoded_corpus = tokenizer.batch_decode(ids, skip_special_tokens=True)
        decoded_corpus = tokenizer.batch_decode(ids, skip_special_tokens=False)
        print(decoded_corpus[0])

        # Getting the length of the corpus in UTF-8 bytes
        l_b = sum(len(seq.encode('utf-8')) for seq in decoded_corpus)

        # Getting the length of the corpus in T5 tokens
        encoded_corpus = tokenizer.batch_encode_plus(decoded_corpus, add_special_tokens=False)
        l_t = sum(len(seq) for seq in encoded_corpus.input_ids)
        print(l_t/l_b)


        bpb = ((l_t / l_b) * args.loss) / np.log(2)

    else:
        bpb = (args.lt_over_lb * args.loss) / np.log(2)

    print('BPB:', bpb)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--spec-file', type=Path, help='JSON spec file.')
    parser.add_argument('--lt-over-lb', type=float)
    parser.add_argument('--loss', type=float)
    args = parser.parse_args()
    main(args)