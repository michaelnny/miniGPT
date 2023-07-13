import os
import argparse
from contextlib import nullcontext
import torch
import torch.distributed as dist

from models import GPT2LMHeadModel
from utils import load_full_state_model_checkpoint, build_gpt2_tokenizer, sample_sequence


def main(args):
    tokenizer = build_gpt2_tokenizer()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    bf16_ready = torch.version.cuda and torch.cuda.is_bf16_supported() and dist.is_nccl_available()

    mp_ctx = (
        nullcontext()
        if device == 'cpu'
        else torch.amp.autocast(device_type=device, dtype=torch.bfloat16 if bf16_ready else torch.float16)
    )

    print('Initialize model ...')

    model = GPT2LMHeadModel(model_type=args.model_type)

    if os.path.exists(args.ckpt_file):
        load_full_state_model_checkpoint(model, 0, args.ckpt_file)

    # # Model quantization
    # model = torch.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)

    model.eval()
    model.to(device)

    stop_token_ids = [tokenizer.eot_token]

    # pretrained model is to predict next token, so we need to be careful how to form the prompt.
    prompts = [
        "Tell me a joke about a dog.",
        "What is the meaning of life?",
        "Explain what is the theory of relativity.",
        "Who is Steve Jobs?",
        "Who is John F. Kennedy?",
        "Who is Kevin Hart?",
        "Who is Michael Jackson?",
        "How to asking for a pay raise?",
        "What is a Put option in finance?",
        "What is a guitar?",
        "How often should people exercise to stay healthy?",
        "When did the first World war start?",
        "Why are most plants green?",
        "How to take care a pet turtle?",
        "Is the following statement true or false: cats can fly?",
        "Is fish live on land?",
        "What language is spoken in Brazil?",
        "What language is spoken in China?",
        "What is the best season to visit United States?",
        "What is the best season to visit Japan?",
        "Explain moon landing in simple words.",
        "If I want to raise a pet, what should I chose, dog or cat?",
        "What's the capital city of Japan?",
        "What's the capital city of United States?",
    ]

    for prompt in prompts:
        print("\n" + "#" * 80 + "\n")
        print(f'Prompt: "{prompt}"')
        print("\n" + "=" * 8 + ">\n")

        context = tokenizer.encode(prompt)
        context = torch.tensor(context, dtype=torch.long, device=device)[None, ...]

        with mp_ctx:
            out = sample_sequence(
                model=model,
                context=context,
                stop_token_ids=stop_token_ids,
                max_gen_seq_length=args.max_gen_seq_length,
                temperature=args.temperature,
                repetition_penalty=args.repetition_penalty,
                top_k=args.top_k,
                top_p=args.top_p,
            )

        out_tokens = out[0].tolist()
        out_text = tokenizer.decode(out_tokens)

        out_text = out_text[: out_text.find('<|endoftext|>')]
        print(f"{out_text.strip()}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test Pre-trained GPT-2 model')

    parser.add_argument('--top_k', type=int, default=200, help='')
    parser.add_argument('--top_p', type=float, default=0.95, help='')
    parser.add_argument('--temperature', type=float, default=0.8, help='')
    parser.add_argument('--repetition_penalty', type=float, default=1.1, help='')
    parser.add_argument('--max_gen_seq_length', type=int, default=500, help='')

    parser.add_argument(
        '--model_type',
        type=str,
        default='gpt2-xl',
        help='gpt2, gpt2-medium, gpt2-large, gpt2-xl, model smaller than gpt2-large performs very poor',
    )
    parser.add_argument('--ckpt_file', type=str, default='./checkpoints/gpt2-xl-openai-pretrained.pt', help='')

    parser.add_argument('--seed', type=int, default=133, help='random seed (default: 1)')
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    main(args)
