"""This script converts the data from CFET format to BFET format.

Usage:
python cfet2bfet.py \
    -i /shared/nas/data/m1/yinglin8/projects/fet/data/aida_2020/en/en.cfet.json \
    -o /shared/nas/data/m1/yinglin8/projects/fet/data/aida_2020/en/en.bfet.json \
    -m roberta-large \
    -c /shared/nas/data/m1/yinglin8/embedding/bert \
    -l 128
"""
import json
from typing import List, Tuple
from argparse import ArgumentParser, Namespace
from transformers import (PreTrainedTokenizerFast,
                          BertTokenizerFast,
                          RobertaTokenizerFast)


def create_tokenizer(model_name: str, cache_dir: str) -> PreTrainedTokenizerFast:
    """Creates a tokenizer given a model name.

    Args:
        model_name (str): model name. Currently only bert-* and roberta-* models
            are supported.
            See: https://huggingface.co/transformers/pretrained_models.html
        cache_dir (str): path to the Bert cache dir. Defaults to None.

    Returns:
        PreTrainedTokenizerFast: a tokenizer that converts words to word pieces.
    """
    if model_name.startswith('bert'):
        tokenizer = BertTokenizerFast.from_pretrained(
            model_name, cache_dir=cache_dir)
    elif model_name.startswith('roberta'):
        tokenizer = RobertaTokenizerFast.from_pretrained(
            model_name, cache_dir=cache_dir)
    else:
        raise ValueError('Unknown model name: {}'.format(model_name))

    return tokenizer


def tokenize(tokens: List[str],
             tokenizer: PreTrainedTokenizerFast
            ) -> Tuple[List[str], List[Tuple[int, int]]]:
    """Converts a token list to a word piece list.

    Args:
        tokens (List[str]): a list of tokens.
        tokenizer (PreTrainedTokenizerFast): a tokenizer that converts tokens
            to word pieces.

    Returns:
        pieces_list_flat (List[str]): a list of word pieces.
        piece_indices (List[Tuple[int, int]]): a list of index tuples. The 
    """
    pieces_list = [tokenizer.tokenize(t) for t in tokens]
    offset = 0
    piece_indices = []
    pieces_list_flat = []
    for pieces in pieces_list:
        end = offset + len(pieces)
        piece_indices.append((offset, end))
        offset = end
        pieces_list_flat.extend(pieces)
    return pieces_list_flat, piece_indices


def convert(input_file: str,
            output_file: str,
            bert_model: str = 'bert-large-cased',
            bert_cache_dir: str = None,
            max_len: int = 128):
    """Converts fine-grained entity typing data from the CFET format to BFET
    format.

    Args:
        input_file (str): path to the CFET format input file.
        output_file (str): path to the BFET format output file.
        bert_model (str, optional): Bert model name. Defaults to
            'bert-large-cased'.
        bert_cache_dir (str, optional): path to the Bert cache folder. Defaults
            to None.
        max_len (int, optional): max sentence length. Defaults to 128.
    """
    # Create the tokenizer
    tokenizer = create_tokenizer(bert_model, bert_cache_dir)
    
    # [CLS] and [SEP] tokens
    text_max_len = max_len - 2
    
    overlength_num = 0
    with open(input_file) as r, open(output_file, 'w') as w:
        for line in r:
            inst = json.loads(line)
            tokens = inst['tokens']
            pieces, piece_indices = tokenize(tokens)
            # Skip overlength examples
            if len(pieces) > text_max_len:
                overlength_num += 1
                continue
            # Convert pieces to piece indices
            inst['pieces'] = tokenizer.encode(pieces,
                                              truncation=True,
                                              max_length=max_len)
            # Add word piece start/end offsets
            for annotation in inst['annotations']:
                annotation['piece_start'] = piece_indices[annotation['start']]
                annotation['piece_end'] = piece_indices[annotation['end']]
            
            w.write(json.dumps(inst) + '\n')


def parse_arguments() -> Namespace:
    """Parses commandline arguments.

    Returns:
        Namespace: a namespace object for parsed arguments.
    """
    parser = ArgumentParser()
    parser.add_argument('-i', '--input', help='Path to the input file.')
    parser.add_argument('-o', '--output', help='Path to the output file.')
    parser.add_argument('-m', '--model_name', help='Bert model name')
    parser.add_argument('-c', '--cache_dir', help='Bert cache directory')
    parser.add_argument('-l', '--max_len', type=int, help='Max sentence length')
    args = parser.parse_args()
    return args


def main():
    args = parse_arguments()

    convert(args.input,
            args.output,
            args.model_name,
            args.cache_dir,
            args.max_len)


if __name__ == '__main__':
    main()