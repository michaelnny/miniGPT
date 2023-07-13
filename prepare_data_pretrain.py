"""
Module for cleanup raw text files before using it to build datasets.
"""

import functools
from concurrent.futures import ProcessPoolExecutor, as_completed
import os
import shutil
import re
import ftfy
import json


from utils import create_logger, find_certain_files_under_dir, read_txt_file, read_jsonl_file, count_words


# File name or book name contains any of these keywords are discarded
# As the BooksCorpus contains too much 'romance' books, we also want to remove anything related religion and other domain
BLACKLIST_WORDS = [
    'romance',
    'romantic',
    'love',
    'fairy',
    'kiss',
    'story',
    'beauty',
    'pretty',
    'girl',
    'tale',
    'evil',
    'devil',
    'hell',
    'vampire',
    'zombie',
    'ghost',
    'monster',
    'shakespeares',
    'bible',
    'chronicles',
    'christianity',
    'god',
    'jesus',
    'holy',
    'church',
    'islam',
    'muslim',
    'sex',
    'sins',
    'sixfold',
    'novel',
    'poems',
    'quantum-troopers',
    'shades-of-gray',
    'slaughter',
    'obama',
    'trump',
    'dowsing',
    'magic',
    '314',
    'dog',
    'multiobjective',
    'torture',
    'faith',
    'divine',
]


URL_PLACEHOLDER = 'URL_HOLDER'
EMAIL_PLACEHOLDER = 'EMAIL_HOLDER'

URL_PATTERN = r"\b(?:(?:https?|ftp)://|www\.)[^@\s]+\b|(?!.*@)\b(?:\w+\.)+(?:com|org|net|edu|cn)\S*"
EMAIL_PATTERN = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'

# Remove some special characters from the text
COMMON_RULES = {
    r'[-—*━一=]+(\n|$)': '\n',  # standardize line breaker patterns like ------------ or ———————————————or ************* or ━━━━━━━━━━━━━━━━━━
    r'[*━]{3,}': '',  # Special cases
    # r'\([,.;:，。；：`‘\s]{1,}': '(',
    # r'[,.;:，。；：`‘\s]{1,}\)': ')',
}

# Define a mapping of punctuation characters to a standardized version for English language
EN_RULES = {
    **COMMON_RULES,
    r"[“”]|[‘’]": '"',  # Replace curly quotes with straight quotes
    r"[`´]": "'",
    r"\.{3,}": '...',
    r"\:,": ',',
    r"\：.": '.',
    r" \'": "'",
    r" \!": '!',
    r" \,": ',',
    r" \.": '.',
    r" \?": '?',
    r" - ": '-',
    r" \)": ')',
    r"\( ": '(',
    r'\(\s*[^\w\s]*\s*\)': '',  # parenthesis without any text inside
    r"(?:(?!\n)\s)+": ' ',  # Normalize whitespace while preserving new lines
    r"[\n]{3,}": '\n\n',  # standardize new lines
}

# Define a mapping of punctuation characters to a standardized version for Chinese language
ZH_RULES = {
    **COMMON_RULES,
    r":": '：',
    r";": '；',
    r'(?<![0-9%.])\,': '，',  # skip if in numbers
    r'(?<![0-9%.])\.': '。',  # BUG this will break ...
    r"\?": '？',
    r"\!": '！',
    r"\。{2,}": '。',
    r"\{1,}": '',
    r"\：，": '，',
    r"\：。": '。',
    r"\“\s*[^\w\s]*\s*\”": '',
    r'\(\s*[^\w\s]*\s*\)': '',  # parenthesis without any text inside
    r'\《\s*[^\w\s]*\s*\》': '',  # chinese version parenthesis without any text inside
    r'\（\s*[^\w\s]*\s*\）': '',
    r'\（\s*[^\w\s]*\s*\)': '',
    r'\(\s*[^\w\s]*\s*\）': '',
    # r'\（[，。；：`‘\s]{1,}': '（',
    # r'[，。；：`‘\s]{1,}\）': '）',
    r'生平。': '',
    r'\(.*縮寫\s*[^\w\s]*\s*\）': '',
    r'\（学名\）': '',
    r'\（学名\s*[^\w\s]*\s*\）': '',
    r'\（學名\）': '',
    r'\（學名\s*[^\w\s]*\s*\）': '',
    r'\（或\）': '',
    r'\（\s或\s\）': '',
    r'\（维吾尔语： / \）': '',
    r'\（邮政式拼音\：\）': '',
    r'\（.*法語：\）': '',
    r'\（.*普通话：\）': '',
    r"(●|■|★)": '-',
    r"(\[空行\]|.*?章节内容开始.*?|.*?本章结束.*?|.*?本章完.*?|_分节阅读_.*?)\n": '\n',
    r"(声明|申明|说明|更新时间)：.*?\n": '',
    r"(.*?网.*?整理|查看.*?书籍推荐请到.*?|小说天堂.*?|更多.*?请浏览.*?|更多.*?请访问.*?|.*?搜刮精品小说.*?|.*?免费下载阅读.*?|.*?图书由.*?整理.*?)\n": '',
    r"(.*?作品来自互联网.*?|.*?不得用作商业用途.*?|小说下载.*|.*?电子书.*?书包网.*?|.*?xiaoshuo.*?|本文由.*?小说.*?下载.*?|www.xia.*?)\n": '',
    r"》TXT": '》',
    r"[\n]{3,}": '\n\n',  # standardize new lines
}

# Precompile regular expressions
EN_PATTERNS = {re.compile(pattern, flags=re.IGNORECASE): replacement for pattern, replacement in EN_RULES.items()}

ZH_PATTERNS = {re.compile(pattern): replacement for pattern, replacement in ZH_RULES.items()}


def handle_hyphenated_words(text):
    """Try to find and fix the end-of-line hyphen mark which divides a single word."""
    processed_lines = []
    merge_next_line = False
    merged_word = ""

    for line in text.splitlines():
        if merge_next_line:
            merged_word += line.strip()
            if line.strip() and line.strip()[0].isalpha():
                processed_lines.append(merged_word)
                merge_next_line = False
                merged_word = ""
        elif line.endswith("-"):
            merge_next_line = True
            merged_word = line[:-1]
        else:
            processed_lines.append(line.strip())

    processed_text = "\n".join(processed_lines)
    return processed_text


def contains_blacklist_keywords(text):
    if text is None or text == '':
        return False
    return any(k in text for k in BLACKLIST_WORDS)


def start_with_invalid_characters(text):
    """Check the start characters of the text, his is mostly for dataset from internet like wikipedia"""
    if text.startswith('<html'):
        return True
    if text.startswith('&lt'):
        return True
    if re.match(r'^[.,\'|"\)}]\s', text):
        return True
    if text.startswith('-{'):
        return True
    if text.startswith('《-{'):
        return True
    if text.startswith('""'):
        return True
    if text.startswith('“”'):
        return True
    if text.startswith('\\'):
        return True
    if text.startswith('{{'):
        return True
    if text.startswith('!colspan='):
        return True
    if text.startswith('! colspan='):
        return True
    if text.startswith('！colspan='):
        return True
    if text.startswith('！ colspan'):
        return True
    if text.startswith('(or)'):
        return True
    return False


def too_much_invalid_characters(text, threshold=5):
    # too much empty parenthesis
    if text.count('{}') >= threshold:
        return True
    if text.count('()') >= threshold:
        return True
    if text.count('-{zh-hans：') >= threshold:
        return True

    return False


def extract_unique_filenames(all_files):
    """Try to remove duplicate or similar files, also remove file if the file name contains unsafe keyword."""
    exits_names = set()

    unique_files = []

    for file in all_files:
        base_name = os.path.basename(file)

        # Convert to lowercase
        base_name = base_name.strip().lower()

        if contains_blacklist_keywords(base_name):
            continue

        # Remove file extensions
        base_name = re.sub(r'\b(epub|txt|jsonl)', '', base_name)
        base_name = re.sub(r'\s', '-', base_name)

        # Remove non-alphanumeric and non-hyphen characters
        base_name = re.sub(r'[^a-zA-Z\-]', '', base_name)
        base_name = re.sub(r"([-]+)", '-', base_name)

        if base_name in exits_names:
            continue

        exits_names.add(base_name)

        unique_files.append(file)

    return unique_files


def cleanup_raw_text(raw_text, preserve_url, preserve_email, is_chinese=False):
    """
    Perform cleanup and standardization tasks including:

    1. Handle bad Unicode using ftfy
    2. Handle end of sentence hyphen
    3. Standardize and preserve punctuations
    4. Normalize white spaces
    """

    if not is_chinese:
        raw_text = ftfy.fix_text(raw_text)
        raw_text = handle_hyphenated_words(raw_text)

    raw_text = raw_text.strip()

    # Find and store URLs and emails on demand
    # It's important we do this first for email and then urls, also need apply before actual clean up tasks,
    # as those simple rules can't handle these cases properly

    emails = re.findall(EMAIL_PATTERN, raw_text)
    raw_text = re.sub(EMAIL_PATTERN, EMAIL_PLACEHOLDER, raw_text)

    urls = re.findall(URL_PATTERN, raw_text)
    raw_text = re.sub(URL_PATTERN, URL_PLACEHOLDER, raw_text)

    re_patterns = EN_PATTERNS
    if is_chinese:
        re_patterns = ZH_PATTERNS

    cleaned_text = raw_text
    for pattern, replacement in re_patterns.items():
        cleaned_text = re.sub(pattern, replacement, cleaned_text)
        cleaned_text = cleaned_text.strip()

    # Restore the URLs and emails after all tasks have been done
    if preserve_url:
        # Use a loop to make sure only the first occurrence of placeholder is replaced with the URL
        for url in urls:
            cleaned_text = cleaned_text.replace(URL_PLACEHOLDER, url, 1)
    else:
        cleaned_text = re.sub(URL_PLACEHOLDER, '', cleaned_text)

    if preserve_email:
        for email in emails:
            cleaned_text = cleaned_text.replace(EMAIL_PLACEHOLDER, email, 1)
    else:
        cleaned_text = re.sub(EMAIL_PLACEHOLDER, '', cleaned_text)

    if start_with_invalid_characters(cleaned_text):
        return ''

    if too_much_invalid_characters(cleaned_text):
        return ''

    return cleaned_text


def cleanup_single_txt_file(
    id,
    input_file,
    output_dir,
    min_words,
    preserve_url=True,
    preserve_email=True,
    is_chinese=False,
):
    """Load a single .txt file and preform cleanup, and saved the cleaned text to output directory with an identical file name."""

    base_name = os.path.basename(input_file)

    # Construct the output file path by joining the output directory and the file name
    output_file = os.path.join(output_dir, f'{id}_{base_name}')

    if contains_blacklist_keywords(base_name):
        return 0

    raw_text = read_txt_file(input_file, is_chinese)

    if raw_text is None:
        return 0

    # Apply cleanup rules to the text
    cleaned_text = cleanup_raw_text(
        raw_text,
        preserve_url=preserve_url,
        preserve_email=preserve_email,
        is_chinese=is_chinese,
    )

    num_words = count_words(cleaned_text, is_chinese)
    if num_words < min_words:
        return 0

    # Save the cleaned text to the output file using utf-8 encoding
    with open(output_file, "w", encoding="utf-8") as new_file:
        new_file.write(cleaned_text)

    return num_words


def cleanup_single_jsonl_file(
    id,
    input_file,
    output_dir,
    min_words,
    preserve_url=True,
    preserve_email=True,
    is_chinese=False,
):
    base_name = os.path.basename(input_file)

    # Construct the output file path by joining the output directory and the file name
    output_file = os.path.join(output_dir, f'{id}_{base_name}')

    # Generator yields a list of json objects
    json_objs = read_jsonl_file(input_file)

    if json_objs is None:
        return 0

    # Open output files
    words_count = 0
    with open(output_file, 'w', encoding='utf-8') as new_file:
        for data in json_objs:
            if not 'text' in data:
                continue

            # Access the text field
            raw_text = data['text']

            # Apply cleanup rules to the text
            cleaned_text = cleanup_raw_text(
                raw_text,
                preserve_url=preserve_url,
                preserve_email=preserve_email,
                is_chinese=is_chinese,
            )

            num_words = count_words(cleaned_text, is_chinese)
            if num_words < min_words:
                continue

            # Write the cleaned JSON object to the output file
            new_file.write(json.dumps({"text": cleaned_text}, ensure_ascii=False if is_chinese else True) + '\n')
            words_count += num_words

    return words_count


def cleanup_files(
    src_dir,
    output_dir,
    file_type,
    logger,
    num_workers=8,
    min_words=100,
    filter_by_fname=False,
    preserve_url=False,
    preserve_email=False,
    overwrite_output=False,
    is_chinese=False,
):
    """Given a source directory, try to find all files matching file_type in this and sub folders.
    For each founded file, perform cleanup, and save the cleaned text to the output_dir with an identical file name.
    Text has lesser words than min_words is discarded.
    """
    assert src_dir != output_dir
    assert os.path.exists(src_dir) and os.path.isdir(src_dir)

    assert file_type in ['.txt', '.jsonl']

    working_files = find_certain_files_under_dir(src_dir, file_type=file_type)

    if filter_by_fname:
        working_files = extract_unique_filenames(working_files)

    num_files = len(working_files)

    if num_files == 0:
        logger.warning(f'Found no {file_type} file under "{src_dir}"')
        return
    else:
        logger.info(f'Found {num_files} {file_type} files under "{src_dir}"')

    if os.path.exists(output_dir):
        if len(os.listdir(output_dir)) != 0:
            # Remove the output directory and all it's content
            if overwrite_output:
                logger.info(f'cleanup output folder "{output_dir}"')
                shutil.rmtree(output_dir)
            else:
                logger.error(f'The output folder "{output_dir}" is not empty')
                return

    # Create the output directory if necessary
    os.makedirs(output_dir, mode=0o777, exist_ok=True)

    logger.info(f'Processing {num_files} {file_type} files using {num_workers} workers ...')

    target_func_name = cleanup_single_txt_file
    if file_type == '.jsonl':
        target_func_name = cleanup_single_jsonl_file

    cleanup_func = functools.partial(
        target_func_name,
        output_dir=output_dir,
        preserve_url=preserve_url,
        preserve_email=preserve_email,
        min_words=min_words,
        is_chinese=is_chinese,
    )

    total_num_words = 0
    last_logged = 0

    # Create a ProcessPoolExecutor with maximum N processes
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(cleanup_func, i, file) for i, file in enumerate(working_files)]

        for future in as_completed(futures):
            word_count = future.result()
            total_num_words += word_count

            if total_num_words - last_logged >= 50e6:
                logger.info(f'Processed {total_num_words/1e6:.2f} million words ...')
                last_logged = total_num_words

    logger.info(f'Finished processing {total_num_words/1e6:.2f} million words')

    cleaned_files = find_certain_files_under_dir(output_dir, file_type=file_type)
    logger.info(f'Found {len(cleaned_files)} cleaned up {file_type} files under "{output_dir}"')


if __name__ == "__main__":
    logger = create_logger()

    cleanup_files(
        src_dir="./raw_data/zhwiki_latest",
        output_dir="./clean_data/zhwiki_latest",
        file_type='.jsonl',
        logger=logger,
        is_chinese=True,
        overwrite_output=False,
        num_workers=20,
    )

    cleanup_files(
        src_dir="./raw_data/enwiki_latest",
        output_dir="./clean_data/enwiki_latest",
        file_type='.jsonl',
        logger=logger,
        overwrite_output=False,
        num_workers=20,
    )

    cleanup_files(
        src_dir="./raw_data/openwebtext2/",
        output_dir="./clean_data/openwebtext2",
        file_type='.jsonl',
        logger=logger,
        preserve_url=True,
        overwrite_output=False,
        num_workers=20,
    )
