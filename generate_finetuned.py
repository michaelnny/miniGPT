import os
import argparse
from contextlib import nullcontext
import torch
import torch.distributed as dist

from models import GPT2LMHeadModel
from utils import load_full_state_model_checkpoint, build_gpt2_tokenizer, build_prompt_completion, sample_sequence, END_TOKEN


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

    stop_token_ids = [tokenizer.eot_token, tokenizer.encode_single_token(END_TOKEN)]

    # general questions
    tasks = [
        {"prompt": "Tell me a joke about a dog.", "context": ""},
        {"prompt": "What is the meaning of life?", "context": ""},
        {"prompt": "Explain what is the theory of relativity.", "context": ""},
        {"prompt": "Who is Steve Jobs?", "context": ""},
        {"prompt": "Who is John F. Kennedy?", "context": ""},
        {"prompt": "Who is Kevin Hart?", "context": ""},
        {"prompt": "Who is Michael Jackson?", "context": ""},
        {"prompt": "How to asking for a pay raise?", "context": ""},
        {"prompt": "What is a Put option in finance?", "context": ""},
        {"prompt": "What is a guitar?", "context": ""},
        {"prompt": "How often should people exercise to stay healthy?", "context": ""},
        {"prompt": "When did the first World war start?", "context": ""},
        {"prompt": "Why are most plants green?", "context": ""},
        {"prompt": "How to take care a pet turtle?", "context": ""},
        {"prompt": "Is the following statement true or false: cats can fly?", "context": ""},
        {"prompt": "Is fish live on land?", "context": ""},
        {"prompt": "What language is spoken in Brazil?", "context": ""},
        {"prompt": "What language is spoken in China?", "context": ""},
        {"prompt": "What is the best season to visit United States?", "context": ""},
        {"prompt": "What is the best season to visit Japan?", "context": ""},
        {"prompt": "Explain moon landing in simple words.", "context": ""},
        {"prompt": "If I want to raise a pet, what should I chose, dog or cat?", "context": ""},
        {"prompt": "What's the capital city of Japan?", "context": ""},
        {"prompt": "What's the capital city of United States?", "context": ""},
    ]

    # # the model performs terrible on math problems
    # tasks = [
    #     {"prompt": "What is 1 + 1?", "context": ""},
    #     {"prompt": "What is 3 + 21?", "context": ""},
    #     {"prompt": "Add -87738 and 29.", "context": ""},  # -87709
    #     {"prompt": "Calculate -2632 divided by -94.", "context": ""},  # 28
    # ]

    # # entity extractions
    # tasks = [
    #     {
    #         "prompt": "Who does the Navy Cross is awarded to?",
    #         "context": "The Navy Cross is the United States Naval Service's second-highest military decoration awarded for sailors and marines who distinguish themselves for extraordinary heroism in combat with an armed enemy force. The medal is equivalent to the Army's Distinguished Service Cross, the Air and Space Forces' Air Force Cross, and the Coast Guard Cross.\n\nThe Navy Cross is bestowed by the Secretary of the Navy and may also be awarded to members of the other armed services, and to foreign military personnel while serving with the U.S. Naval Service. The Navy Cross was established by Act of Congress (Public Law 65-253) and approved on February 4, 1919.",
    #     },
    #     {
    #         "prompt": "Based on this paragraph on Japanese bullet trains, how many cars do the longest trains have?",
    #         "context": "Trains are up to sixteen cars long. With each car measuring 25 m (82 ft) in length, the longest trains are 400 m (1\u20444 mile) end to end. Stations are similarly long to accommodate these trains. Some of Japan's high-speed maglev trains are considered Shinkansen, while other slower maglev trains (such as the Linimo maglev train line serving local community near the city of Nagoya in Aichi, Japan) are intended as alternatives to conventional urban rapid transit systems.",
    #     },
    #     {
    #         "prompt": "Who is the largest coatings company in the world by revenue?",
    #         "context": "PPG Industries, Inc. is an American Fortune 500 company and global supplier of paints, coatings, and specialty materials. With headquarters in Pittsburgh, Pennsylvania, PPG operates in more than 70 countries around the globe. By revenue it is the largest coatings company in the world followed by AkzoNobel. It is headquartered in PPG Place, an office and retail complex in downtown Pittsburgh, and is known for its glass facade designed by Postmodern architect Philip Johnson.",
    #     },
    #     {
    #         "prompt": "Give me a summary of Dataphor based on this text.",
    #         "context": "Dataphor is an open-source truly-relational database management system (RDBMS) and its accompanying user interface technologies, which together are designed to provide highly declarative software application development. The Dataphor Server has its own storage engine or it can be a virtual, or federated, DBMS, meaning that it can utilize other database engines for storage.\n\nDataphor has been praised for its adherence to relational principles, more closely so than any SQL product.",
    #     },
    #     {
    #         "prompt": "What does the Gini coefficient measure?",
    #         "context": "In economics, the Gini coefficient, also known as the Gini index or Gini ratio, is a measure of statistical dispersion intended to represent the income inequality or the wealth inequality or the consumption inequality within a nation or a social group. It was developed by statistician and sociologist Corrado Gini.\nThe Gini coefficient measures the inequality among values of a frequency distribution, such as levels of income. A Gini coefficient of 0 reflects perfect equality, where all income or wealth values are the same, while a Gini coefficient of 1 (or 100%) reflects maximal inequality among values. For example, if everyone has the same income, the Gini coefficient will be 0. In contrast, a Gini coefficient of 1 indicates that within a group of people, a single individual has all the income or consumption, while all others have none.\nThe Gini coefficient was proposed by Corrado Gini as a measure of inequality of income or wealth.  For OECD countries, in the late 20th century, considering the effect of taxes and transfer payments, the income Gini coefficient ranged between 0.24 and 0.49, with Slovenia being the lowest and Mexico the highest. African countries had the highest pre-tax Gini coefficients in 2008\u20132009, with South Africa having the world's highest, estimated to be 0.63 to 0.7, although this figure drops to 0.52 after social assistance is taken into account, and drops again to 0.47 after taxation. The global income Gini coefficient in 2005 has been estimated to be between 0.61 and 0.68 by various sources.",
    #     },
    # ]

    for task in tasks:
        print("\n" + "-" * 60 + "\n")

        if len(task["context"]) > 0:
            print(f'Context: {task["context"]}\n')

        print(f'Prompt: {task["prompt"]}\n')

        formatted_prompt, _ = build_prompt_completion(prompt=task["prompt"], completion="None", context=task["context"])
        context = tokenizer.encode(formatted_prompt)
        start_idx = len(context)
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
        out_text = tokenizer.decode(out_tokens[start_idx:])

        out_text = out_text[: out_text.find(END_TOKEN)]
        out_text = out_text[: out_text.find('###')]
        print(f"{out_text.strip()}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test fine-tuned GPT-2 model')

    parser.add_argument('--top_k', type=int, default=200, help='')
    parser.add_argument('--top_p', type=float, default=0.95, help='')
    parser.add_argument('--temperature', type=float, default=0.8, help='')
    parser.add_argument('--repetition_penalty', type=float, default=1.0, help='')
    parser.add_argument('--max_gen_seq_length', type=int, default=500, help='')

    parser.add_argument(
        '--model_type',
        type=str,
        default='gpt2-xl',
        help='gpt2, gpt2-medium, gpt2-large, gpt2-xl, model smaller than gpt2-large performs very poor',
    )
    parser.add_argument('--ckpt_file', type=str, default='./checkpoints/gpt2-xl-finetune-iter-4000-merged.pt', help='')

    parser.add_argument('--seed', type=int, default=133, help='random seed (default: 1)')
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    main(args)
