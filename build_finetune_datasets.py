import os
import shutil
import random
import json
import pickle
import copy
import numpy as np

from utils import create_logger, find_certain_files_under_dir, read_jsonl_file, build_gpt2_tokenizer


def build_dataset_from_jsonl_file(
    src_dir,
    output_dir,
    logger,
    max_length=1024,  # prompt + completion lengths greater than this are discarded
    overwrite_output=False,
    metadata={},
    seed=1,
):
    """We assumes the files have already been pre-processed"""
    assert src_dir != output_dir
    assert os.path.exists(src_dir) and os.path.isdir(src_dir)
    assert 512 <= max_length

    random.seed(int(seed))

    if metadata is None:
        metadata = {}

    working_files = find_certain_files_under_dir(src_dir, file_type='.jsonl')

    num_files = len(working_files)
    if num_files == 0:
        logger.warning(f'Found no .jsonl file under "{src_dir}", aborting...')
        return

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
    metadata['data_structure'] = 'A list of prompt:completion token sequences pairs. '

    logger.info(metadata)

    logger.info(f'Start to processing {num_files} .jsonl files, this may take a minute...')

    for file in working_files:
        # store training sample pairs
        XY_pairs = []

        # Generator yields a list of json objects
        json_objs = read_jsonl_file(file)

        if json_objs is None:
            continue

        for data in json_objs:
            if 'prompt' not in data or 'completion' not in data:
                continue

            x = data['prompt']
            y = data['completion']

            x_tokens, y_tokens = tokenizer.encode(x), tokenizer.encode(y)

            # skip samples greater than the max length
            if len(x_tokens) + len(y_tokens) > max_length:
                continue

            XY_pairs.append((x_tokens, y_tokens))

        # write to target folder
        base_name = os.path.splitext(os.path.basename(file))[0]
        dataset_file = os.path.join(output_dir, f'{base_name}.pkl')

        logger.info(f'Saving dataset to "{dataset_file}" ...')
        pickle.dump(XY_pairs, open(dataset_file, 'wb'))

        meta_file_json = os.path.join(output_dir, f'{base_name}_meta.json')
        meta_file_kpl = os.path.join(output_dir, f'{base_name}_meta.pkl')
        logger.info(f'Saving metadata to "{meta_file_json}" and "{meta_file_kpl}" ...')

        meta = copy.deepcopy(metadata)
        meta['num_samples'] = len(XY_pairs)
        seq_length = [len(x) + len(y) for x, y in XY_pairs]
        meta['num_tokens'] = sum(seq_length)
        meta['sequence_length_stats'] = {
            'min': int(np.min(seq_length)),
            'max': int(np.max(seq_length)),
            'mean': int(np.mean(seq_length)),
            'std': int(np.std(seq_length)),
        }

        # Also save metadata to .pkl since json file can be easily changed
        pickle.dump(meta, open(meta_file_kpl, 'wb'))

        with open(meta_file_json, 'w', encoding='utf-8') as f:
            json.dump(meta, f, indent=2, sort_keys=True)


if __name__ == '__main__':
    logger = create_logger()

    build_dataset_from_jsonl_file(
        src_dir='./clean_data/SQuAD',
        output_dir='./datasets/SQuAD',
        logger=logger,
        overwrite_output=False,
        metadata={'name': 'SQuAD', 'language': 'English', 'home_page': 'https://rajpurkar.github.io/SQuAD-explorer/'},
    )

    build_dataset_from_jsonl_file(
        src_dir='./clean_data/MARCO_QnA',
        output_dir='./datasets/MARCO_QnA',
        logger=logger,
        overwrite_output=False,
        metadata={'name': 'MARCO QnA', 'language': 'English', 'home_page': 'https://microsoft.github.io/msmarco/'},
    )

    build_dataset_from_jsonl_file(
        src_dir='./clean_data/dolly',
        output_dir='./datasets/dolly',
        logger=logger,
        overwrite_output=False,
        metadata={
            'name': 'Dolly',
            'language': 'English',
            'home_page': 'https://www.databricks.com/blog/2023/04/12/dolly-first-open-commercially-viable-instruction-tuned-llm',
        },
    )

    build_dataset_from_jsonl_file(
        src_dir='./clean_data/commonsense_dialogues',
        output_dir='./datasets/commonsense_dialogues',
        logger=logger,
        overwrite_output=False,
        metadata={
            'name': 'Commonsense dialogues',
            'language': 'English',
            'home_page': 'https://github.com/alexa/Commonsense-Dialogues',
        },
    )

    build_dataset_from_jsonl_file(
        src_dir='./clean_data/mathematics_dataset_v1.0',
        output_dir='./datasets/mathematics_dataset_v1.0',
        logger=logger,
        overwrite_output=False,
        metadata={
            'name': 'DeepMind Mathematics',
            'language': 'English',
            'home_page': 'https://github.com/deepmind/mathematics_dataset',
        },
    )
