"""
Module for build datasets by loading .txt or .jsonl files and tokenize the content, 
note this module assumes the text have already been cleaned.
"""

import functools
from concurrent.futures import ProcessPoolExecutor, as_completed
import os
import shutil
import random
import json
import pickle
import copy
import numpy as np


from utils import create_logger, find_certain_files_under_dir, read_txt_file, read_jsonl_file, build_gpt2_tokenizer


DATASET_PREFIXES = ['train', 'test', 'eval']


def txt_file_to_tokens(input_file, min_length):
    assert os.path.exists(input_file) and os.path.isfile(input_file)

    raw_text = read_txt_file(input_file)

    if raw_text is None:
        return []

    tokenizer = build_gpt2_tokenizer()
    # Skip special tokens if found in the raw text
    tokens = tokenizer.encode_ordinary(raw_text)

    if len(tokens) < min_length:
        return []

    # Mark end of text
    tokens.append(tokenizer.eot_token)
    return tokens


def jsonl_file_to_tokens(input_file, output_dir, min_length, test_ratio, eval_ratio):
    assert os.path.exists(input_file) and os.path.isfile(input_file)
    assert 0 <= eval_ratio <= 0.2
    assert 0 <= test_ratio <= 0.2
    assert 0 <= min_length <= 500

    # Generator yields a list of json objects
    json_objs = read_jsonl_file(input_file)

    if json_objs is None:
        return []

    tokenizer = build_gpt2_tokenizer()

    # We need to temporarily save the tokens in smaller batches to disk to avoid ran out of RAM
    # and we need to do it per single .jsonl file, since one file could be very large
    temp_files = {}
    current_batches = {}
    num_tokens = {}

    for prefix in DATASET_PREFIXES:
        temp_files[prefix] = []
        current_batches[prefix] = []
        num_tokens[prefix] = 0

    base_name = os.path.basename(input_file)

    for data in json_objs:
        if not 'text' in data:
            continue

        # Access the text field
        raw_text = data['text']

        # Skip special tokens if found in the raw text
        tokens = tokenizer.encode_ordinary(raw_text)

        if len(tokens) < min_length:
            continue

        # Mark end of text
        tokens.append(tokenizer.eot_token)

        ds_prefix = ''
        # Randomly adds the tokens to one of the datasets based on the ratio
        random_v = random.random()
        if random_v < test_ratio + eval_ratio:
            if random_v < test_ratio:
                ds_prefix = 'test'
            else:
                ds_prefix = 'eval'
        else:
            ds_prefix = 'train'

        assert ds_prefix in DATASET_PREFIXES

        assert tokens[-1] == tokenizer.eot_token

        current_batches[ds_prefix].extend(tokens)
        num_tokens[ds_prefix] += len(tokens)

        # Write current batch to disk to free up RAM.
        # one integer takes 28 bytes, so 1GB RAM = 1e9 bytes
        if len(current_batches[ds_prefix]) * 28 >= 2e9:
            assert current_batches[ds_prefix][-1] == tokenizer.eot_token
            # need to avoid collision
            temp_f = os.path.join(output_dir, f'{ds_prefix}_{base_name}_{num_tokens[ds_prefix]}.tmp')
            pickle.dump(current_batches[ds_prefix], open(temp_f, 'wb'))
            temp_files[ds_prefix].append(temp_f)
            current_batches[ds_prefix] = []

    # Handle last batches
    for ds_prefix in DATASET_PREFIXES:
        if len(current_batches[ds_prefix]) > 0:
            assert current_batches[ds_prefix][-1] == tokenizer.eot_token
            # need to avoid collision
            temp_f = os.path.join(output_dir, f'{ds_prefix}_{base_name}_{num_tokens[ds_prefix]}.tmp')
            pickle.dump(current_batches[ds_prefix], open(temp_f, 'wb'))
            temp_files[ds_prefix].append(temp_f)
            current_batches[ds_prefix] = []

    return temp_files, num_tokens


def _merge_and_write_to_disk(temp_files, data_type, max_items, output_file, delete_temp_files=True):
    """
    Combining different temp files together and save the content into a single numpy memmap array.
    """
    assert data_type in [np.uint16, np.uint32]
    assert not os.path.exists(output_file)
    assert max_items > 0

    if len(temp_files) == 0:
        return

    num_added = 0

    memmap_array = np.memmap(output_file, dtype=data_type, mode='w+', shape=(max_items,))

    # Load each of temp files and write to the memmap array
    for f in temp_files:
        try:
            data = pickle.load(open(f, 'rb'))
            start = num_added
            end = min(start + len(data), max_items)

            memmap_array[start:end] = data[:end]
            num_added += len(data)
            if num_added >= max_items or end >= max_items:
                break
        except Exception:
            continue

    # Explicitly flush the file buffer to ensure data is written to disk
    memmap_array.flush()

    # Delete temp files
    if delete_temp_files:
        for f in temp_files:
            os.remove(f)


def _save_dataset_to_disk(metadata, output_dir, data_type, dataset_prefix, num_tokens, temp_files, logger, shuffle=True):
    if shuffle:
        random.shuffle(temp_files)

    save_fname = os.path.join(output_dir, f'{dataset_prefix}.npy')
    logger.info(f'Merging and saving dataset to "{save_fname}" ...')
    _merge_and_write_to_disk(temp_files, data_type, num_tokens, save_fname)

    metadata['num_tokens'] = num_tokens

    meta_file_json = os.path.join(output_dir, f'{dataset_prefix}_meta.json')
    meta_file_kpl = os.path.join(output_dir, f'{dataset_prefix}_meta.pkl')
    logger.info(f'Saving metadata to "{meta_file_json}" and "{meta_file_kpl}" ...')

    # Also save metadata to .pkl since json file can be easily changed
    pickle.dump(metadata, open(meta_file_kpl, 'wb'))

    with open(meta_file_json, 'w', encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, sort_keys=True)


def _build_single_books_dataset(
    working_files, min_length, num_workers, metadata, output_dir, data_type, dataset_prefix, logger
):
    """Given a list of .txt files, build a dataset by tokenize and concatenate the token sequences."""

    assert dataset_prefix is not None and len(dataset_prefix) > 0
    assert min_length is not None and min_length > 0
    assert os.path.exists(output_dir) and os.path.isdir(output_dir)

    num_files = len(working_files)

    if num_files == 0:
        return []

    tokenizer = build_gpt2_tokenizer()
    num_tokens = 0

    # We need to temporarily save the tokens in smaller batches to disk to avoid ran out of RAM
    temp_files = []
    current_batch = []

    # For logging only
    last_logged = 0

    to_tokens_func = functools.partial(txt_file_to_tokens, min_length=min_length)
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(to_tokens_func, file) for file in working_files]

        for i, future in enumerate(as_completed(futures)):
            tokens = future.result()
            assert tokens[-1] == tokenizer.eot_token

            current_batch.extend(tokens)
            num_tokens += len(tokens)

            if num_tokens - last_logged >= 1e8:
                logger.info(f'Processed {num_tokens/1e6:.2f} million tokens ...')
                last_logged = num_tokens

            # Write current batch to disk to free up RAM.
            # one integer takes 28 bytes, so 1GB RAM = 1e9 bytes
            if len(current_batch) * 28 >= 2e9 or (i == len(futures) - 1 and len(current_batch) > 0):
                assert current_batch[-1] == tokenizer.eot_token
                temp_f = os.path.join(output_dir, f'{dataset_prefix}_{num_tokens}.tmp')
                pickle.dump(current_batch, open(temp_f, 'wb'))
                temp_files.append(temp_f)
                current_batch.clear()

    logger.info(f'Finished processing {num_tokens/1e6:.2f} million tokens')

    _save_dataset_to_disk(metadata, output_dir, data_type, dataset_prefix, num_tokens, temp_files, logger)


def build_datasets_from_txt_files(
    src_dir,
    output_dir,
    logger,
    min_length=300,
    eval_ratio=0.1,
    test_ratio=0.0,
    num_workers=8,
    overwrite_output=False,
    metadata={},
    seed=1,
):
    """
    Build pre-train datasets from .txt books.

    This function does the following tasks in sequential order:

        1. Given a source directory, try to find all .txt files in this and sub folders.
        2. Split the files into separate lists for training, test, and evaluation datasets based on the given ratios.
        3. For each of the file lists corresponding to the dataset, build a single stream of token sequence
            by tokenize the different raw texts and then concatenate them together,
            where we use special token <|endoftext|> to separate the document.
        4. Save the dataset to the output folder using np.menmap structure.

    """

    assert src_dir != output_dir
    assert os.path.exists(src_dir) and os.path.isdir(src_dir)
    assert 0 <= eval_ratio <= 0.2
    assert 0 <= test_ratio <= 0.2
    assert 100 <= min_length <= 500

    random.seed(int(seed))

    if metadata is None:
        metadata = {}

    working_files = find_certain_files_under_dir(src_dir, file_type='.txt')

    num_files = len(working_files)
    if num_files == 0:
        logger.warning(f'Found no .txt file under "{src_dir}"')
        return

    logger.info(f'Found {num_files} .txt files under "{src_dir}"')

    if os.path.exists(output_dir):
        if len(os.listdir(output_dir)) != 0:
            # Remove the output directory and all it's content
            if overwrite_output:
                logger.info(f'Cleanup output folder "{output_dir}"')
                shutil.rmtree(output_dir)
            else:
                logger.error(f'The output folder "{output_dir}" is not empty')
                return

    # Create the output directory if necessary
    os.makedirs(output_dir, mode=0o777, exist_ok=True)

    tokenizer = build_gpt2_tokenizer()
    vocab_size = tokenizer.n_vocab
    data_type = np.uint16 if vocab_size < 2**16 else np.uint32

    metadata['tokenizer'] = f'tiktoken-{tokenizer.name}'
    metadata['vocab_size'] = vocab_size
    metadata['data_type'] = np.dtype(data_type).name
    metadata[
        'data_structure'
    ] = 'A stream of token sequences created by combining various tokenized texts together, where document boundary is separated by <|endoftext|>'

    logger.info(metadata)

    # For books corpus, one book is one .txt file
    # So we split the list of file names to build train, test, and evaluation datasets
    logger.info(f'Splitting {num_files} .txt files to training, test, and evaluation datasets')
    random.shuffle(working_files)
    test_idx = int(num_files * test_ratio)
    eval_idx = int(num_files * eval_ratio)

    test_files = working_files[:test_idx] if test_idx > 0 else []
    eval_files = working_files[test_idx : test_idx + eval_idx] if eval_idx > 0 else []
    train_files = working_files[test_idx + eval_idx :]

    build_dataset_func = functools.partial(
        _build_single_books_dataset,
        min_length=min_length,
        num_workers=num_workers,
        output_dir=output_dir,
        data_type=data_type,
        metadata=copy.deepcopy(metadata),
        logger=logger,
    )

    # Build train, test, and evaluation datasets
    if len(train_files) > 0:
        logger.info(f'Processing {len(train_files)} .txt files for training dataset ...')
        build_dataset_func(working_files=train_files, dataset_prefix='train')

    if len(eval_files) > 0:
        logger.info(f'Processing {len(eval_files)} .txt files for evaluation dataset ...')
        build_dataset_func(working_files=eval_files, dataset_prefix='eval')

    if len(test_files) > 0:
        logger.info(f'Processing {len(test_files)} .txt files for test dataset ...')
        build_dataset_func(working_files=test_files, dataset_prefix='test')


def build_datasets_from_jsonl_files(
    src_dir,
    output_dir,
    logger,
    min_length=300,
    eval_ratio=0.1,
    test_ratio=0.0,
    num_workers=8,
    overwrite_output=False,
    metadata={},
    seed=1,
):
    """
    Build pre-train datasets from .jsonl format like openwebtext, wiki etc..

    This function does the following tasks in sequential order:
        1. Given a source directory, try to find all .jsonl files in this and sub folders.
        2. For each of the .jsonl files, read each object (or line) and
            randomly assigned it to training, test, and evaluation datasets based on the given ratios.
            This is done by build a single stream of token sequence
            by tokenize the different raw texts and then concatenate them together,
            where we use special token <|endoftext|> to separate the document.
        3. Save the dataset to the output folder using np.menmap structure.

    """

    assert src_dir != output_dir
    assert os.path.exists(src_dir) and os.path.isdir(src_dir)
    assert 0 <= eval_ratio <= 0.2
    assert 0 <= test_ratio <= 0.2
    assert 100 <= min_length <= 500

    random.seed(int(seed))

    if metadata is None:
        metadata = {}

    working_files = find_certain_files_under_dir(src_dir, file_type='.jsonl')

    num_files = len(working_files)
    if num_files == 0:
        logger.warning(f'Found no .jsonl file under "{src_dir}", aborting...')
        return

    logger.info(f'Found {num_files} .jsonl files under "{src_dir}"')

    if os.path.exists(output_dir):
        if len(os.listdir(output_dir)) != 0:
            # Remove the output directory and all it's content
            if overwrite_output:
                logger.info(f'Cleanup output folder "{output_dir}"')
                shutil.rmtree(output_dir)
            else:
                logger.error(f'The output folder "{output_dir}" is not empty')
                return

    # Create the output directory if necessary
    os.makedirs(output_dir, mode=0o777, exist_ok=True)

    tokenizer = build_gpt2_tokenizer()
    vocab_size = tokenizer.n_vocab
    data_type = np.uint16 if vocab_size < 2**16 else np.uint32

    metadata['tokenizer'] = f'tiktoken-{tokenizer.name}'
    metadata['vocab_size'] = vocab_size
    metadata['data_type'] = np.dtype(data_type).name
    metadata[
        'data_structure'
    ] = 'A stream of token sequences created by combining various tokenized texts together, where document boundary is separated by <|endoftext|>'

    logger.info(metadata)

    # Build train, test, and evaluation datasets all at the same time,
    # because for datasets like wikipedia or openwebtext, one single file may contains lots of texts
    random.shuffle(working_files)

    tokenizer = build_gpt2_tokenizer()

    # We need to temporarily save the tokens in smaller batches to disk to avoid ran out of RAM
    all_temp_files = {}
    all_num_tokens = {}

    for prefix in DATASET_PREFIXES:
        all_temp_files[prefix] = []
        all_num_tokens[prefix] = 0

    # For logging only
    total_num_tokens = 0
    last_logged = 0

    to_tokens_func = functools.partial(
        jsonl_file_to_tokens,
        output_dir=output_dir,
        min_length=min_length,
        eval_ratio=eval_ratio,
        test_ratio=test_ratio,
    )
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(to_tokens_func, file) for file in working_files]

        for future in as_completed(futures):
            result = future.result()
            temp_files, num_tokens = result

            for ds_prefix in DATASET_PREFIXES:
                if len(temp_files[ds_prefix]) > 0:
                    all_temp_files[ds_prefix].extend(temp_files[ds_prefix])
                    all_num_tokens[ds_prefix] += num_tokens[ds_prefix]

            total_num_tokens = sum(all_num_tokens.values())

            if total_num_tokens - last_logged >= 1e8:
                logger.info(f'Processed {total_num_tokens/1e6:.2f} million tokens ...')
                last_logged = total_num_tokens

    logger.info(f'Finished processing {total_num_tokens/1e6:.2f} million tokens')

    # For each dataset, combining different temp files together and save the content into a single numpy memmap array
    for ds_prefix in DATASET_PREFIXES:
        if len(all_temp_files[ds_prefix]) > 0:
            logger.info(f'Saving {ds_prefix} dataset ...')
            _save_dataset_to_disk(
                metadata=copy.deepcopy(metadata),
                output_dir=output_dir,
                data_type=data_type,
                dataset_prefix=ds_prefix,
                num_tokens=all_num_tokens[ds_prefix],
                temp_files=all_temp_files[ds_prefix],
                logger=logger,
            )


if __name__ == "__main__":
    logger = create_logger()

    build_datasets_from_jsonl_files(
        src_dir="./clean_data/zhwiki_latest",
        output_dir="./datasets/zhwiki",
        logger=logger,
        overwrite_output=False,
        num_workers=16,
        metadata={'name': 'zhwiki_202307', 'language': 'Chinese'},
    )

    build_datasets_from_jsonl_files(
        src_dir="./clean_data/enwiki_latest",
        output_dir="./datasets/enwiki",
        logger=logger,
        overwrite_output=False,
        num_workers=16,
        metadata={'name': 'enwiki_202307', 'language': 'English'},
    )

    build_datasets_from_jsonl_files(
        src_dir="./clean_data/openwebtext2",
        output_dir="./datasets/openwebtext2",
        logger=logger,
        overwrite_output=False,
        num_workers=16,
        metadata={'name': 'openwebtext2', 'language': 'English'},
    )
