import tiktoken


def build_gpt2_tokenizer():
    return tiktoken.get_encoding('gpt2')


def build_gpt3x_tokenizer():
    return tiktoken.get_encoding('p50k_base')


def build_gpt4_tokenizer():
    return tiktoken.get_encoding('cl100k_base')


def build_custom_tokenizer(encode_name='p50k_base', special_token_list=[]):
    encode_prefix = encode_name.split('_')[0]
    custom_encode_name = encode_prefix + '_custom'
    enc_base = tiktoken.get_encoding(encode_name)

    max_tk_value = enc_base.n_vocab

    # By default, it already contain the special token <|endoftext|>
    special_tokens = {**enc_base._special_tokens}

    # Add custom special tokens if desired
    for i, tk in enumerate(special_token_list):
        special_tokens[f"<|{tk}|>"] = max_tk_value + (1 + i)

    return tiktoken.Encoding(
        name=custom_encode_name,
        pat_str=enc_base._pat_str,
        mergeable_ranks=enc_base._mergeable_ranks,
        special_tokens=special_tokens,
    )


# if __name__ == '__main__':
#     tokenizer = build_custom_tokenizer(special_token_list=['pad', 'mask']])
#     print(tokenizer.encode('This <|mask|> good.<|endoftext|><|pad|><|mask|><|pad|>', allowed_special='all'))
