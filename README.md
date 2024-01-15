# miniGPT

Try to implement a minimum version of GPT model for research and education purpose. Although we focus on the GPT model, the procedure is suitable to training any auto regressive language models.

**Note**:
Our initial goal was trying to build a full pipeline including RLHF module following the InstructGPT paper, however we soon realized that the pre-trained GPT-2 model is inadequate for complex tasks like answering open-domain questions due to it's small model size. You can checkout the our recent project based on LLaMA at [InstructLLaMA](https://github.com/michaelnny/InstructLLaMA)

## What we got

- data-preprocessing, pre-training, fine-tuning scripts for GPT-2 model
- support PyTorch FSDP for distributed training
- support fine-tuning from loading OpenAI pre-trained weights

# Environment and Requirements

- Python 3.10.6
- PyTorch 2.0.1
- Tensorboard 2.13.0

# Code Structure

- `configs` directory contains all the training configurations like model type, data source, number of iterations, learning rate etc.
- `models` directory contains GPT-2 model definition.
- `utils` directory contains helper modules like checkpoint, logging, tokenization etc.
- `prepare_data_pretrain.py` contains code to clean up raw text before build pre-training datasets.
- `build_pretrain_datasets.py` contains code to build pre-train datasets (tokenize, and them save the dataset in Numpy memmap structure).
- `prepare_data_finetune.py` contains code to process raw data before build fine-tuning datasets.
- `build_finetune_datasets.py` contains code to build fine-tuning datasets (tokenize, and them save the dataset to .jsonl files).
- `custom_dataset.py` contains code for custom dataset instances for pre-training and fine-tuning.
- `pretrain.py` contains code to run pre-training using FSDP.
- `finetune.py` contains code to run full fine-tuning using FSDP.
- `generate_pretrained.py` contains code to evaluate the pre-trained model (predict next token).
- `generate_finetuned.py` contains code to evaluate the fine-tuned model (answer questions).
- `convert_hf_checkpoint.py` contains code to convert OpenAI pre-trained GPT-2 weights to support our model, so we can load it to start fine-tuning.

# Project Setup

```
python3 -m pip install --upgrade pip setuptools

python3 -m pip install -r requirements.txt
```

# Download and prepare datasets

You need download the source files for the individual dataset from the Internet, then using our data preparation and build dataset scripts to turn them into ready to use datasets. We don't provide any ready to use dataset files. Our simple yet not perfect data preparation scripts should be able to handle most common datasets formats (.txt, .json, .jsonl files) with minimum changes.

Once you have downloaded the source files, we can follow these two process to build the training datasets:

- stage 1: raw text files (.txt, .json, .json files) are pre-processed and cleaned up using `prepare_data_pretrain.py` and `prepare_data_finetune.py` scripts.
- stage 2: cleaned up text is then turned into token ids using the `build_pretrain_datasets.py` and `build_finetune_datasets.py` scripts.

## Wikipedia datasets

We can use wikiextractor to extract the rax text from Wikipedia dumps, so later we can use the script to prepare and clean up the text.

```
wikiextractor  ~/Downloads/enwiki-latest-pages-articles.xml.bz2 -b 200M --no-templates --processes 24 --json -o ./raw_data/enwiki_latest

# add .json file extension
find ./raw_data/enwiki_latest -type f -name 'wiki*' -exec sh -c 'x="{}"; mv "$x" "${x}.jsonl"' \;

```

# Pre-training

Once you have the dataset processed and tokenized, you can kick start the pre-training script to pre-train a GPT-2 model. Please notice pre-training will take quite long time and large amount GPU compute power, and we can't use other parameter efficient methods like LoRA during this phase. Unless you have specific reason to do so, we suggest to skip this and using the pre-trained weights from OpenAI instead.

This examples shows how to lunch the pre-training script on a machine with 4 GPUs. By default, the script will write the logs to `./logs/pretrain`.

```
torchrun --standalone --nproc_per_node 4 pretrain.py
```

We can then monitoring the progress by using Tensorboard:

```
tensorboard --logdir=./logs/pretrain
```

# Fine-tuning

Once we have a pre-trained model and the datasets are ready, we can start doing fine-tuning. Note we can skip the pretraining step and using the pretrained model provided by openAI instead (by running the convert_hf_checkpoint.py script).

## Full fine-tuning

Full scale fine-tuning without frozen any parameters. We can start the full scale fine-tuning by either using the weights saved from our pre-training script, or we can load the weights from OpenAI (created by using the convert_hf_checkpoint.py script).

This examples shows how to lunch the full fine-tuning script on a machine with 4 GPUs. By default, the script will write the logs to `./logs/finetune`.

```
torchrun --standalone --nproc_per_node 4 finetune.py
```

We can then monitoring the progress by using Tensorboard:

```
tensorboard --logdir=./logs/finetune
```

# Generate

We have provided some sample tasks in the `generate_pretrained.py` and `generate_finetuned.py` to get you started. Note that the pre-trained model only predicts the next token, so it will often unable to answer to prompt such as general question.

```
python generate_pretrained.py


python generate_finetuned.py
```

The following are some examples from the fine-tuned model (based on OpenAI pretrained gpt2-xl). We can clearly see that the fine-tuned model can answer some (but not all) questions in a reasonably manner. Model size plays a pretty important role here, as we observed, for general question answering tasks, we should use at least the 774M parameters version `gpt2-large`.

```
------------------------------------------------------------
Prompt: Tell me a joke about a dog.

A dog eats a dog.
------------------------------------------------------------

Prompt: What is the meaning of life?

To eat the cookies.

------------------------------------------------------------

Prompt: Explain what is the theory of relativity.

It is a theory of gravity and time, it states that objects are not in a fixed position, but move around the universe at a constant speed.

------------------------------------------------------------

Prompt: Who is Steve Jobs?

Steve Jobs was the co-founder and CEO of Apple.

------------------------------------------------------------

Prompt: Who is John F. Kennedy?

John F. Kennedy was the 35th President of the United States.

------------------------------------------------------------

Prompt: Who is Kevin Hart?

He is a comedian and actor who has appeared in many films and television shows.

------------------------------------------------------------

Prompt: Who is Michael Jackson?

Michael Jackson is a singer. He is the only person in the world to have one of the highest earnings ever.

------------------------------------------------------------

Prompt: How to asking for a pay raise?

You have to ask your manager directly.

------------------------------------------------------------

Prompt: What is a Put option in finance?

A Put option is a bet that you will sell an asset that you own, at a certain price. With a Put option, you risk a fixed amount, and if the price of the asset goes down, you make money. For example, if you own stocks, and the stock price goes down, you make money. If the stock price goes up, you lose money.

------------------------------------------------------------

Prompt: What is a guitar?

It's a pretty cool tool.

------------------------------------------------------------

Prompt: How often should people exercise to stay healthy?

Exercise is important for staying healthy. Exercise in moderation.

------------------------------------------------------------

Prompt: When did the first World war start?

The first World war started in 1914.

------------------------------------------------------------

Prompt: Why are most plants green?

Because they're carbon-rich!

------------------------------------------------------------

Prompt: How to take care a pet turtle?

I would recommend keeping the turtle in a small aquarium with a small amount of regular fish food. I would not recommend keeping it out in the wild, where it could get into problems.

------------------------------------------------------------

Prompt: Is the following statement true or false: cats can fly?

Cats can fly.

------------------------------------------------------------

Prompt: Is fish live on land?

Yes.

------------------------------------------------------------

Prompt: What language is spoken in Brazil?

Portuguese.

------------------------------------------------------------

Prompt: What language is spoken in China?

English.

------------------------------------------------------------

Prompt: What is the best season to visit United States?

summer!

------------------------------------------------------------

Prompt: What is the best season to visit Japan?

It depends on where you want to go, but I would recommend visiting in June or July.

------------------------------------------------------------

Prompt: Explain moon landing in simple words.

The moon landing happened when Richard Nixon, as president of the United States, decided to land a man on the moon.

------------------------------------------------------------

Prompt: If I want to raise a pet, what should I chose, dog or cat?

It's up to you. But they can be very funny to watch.

------------------------------------------------------------

Prompt: What's the capital city of Japan?

Tokyo.

------------------------------------------------------------

Prompt: What's the capital city of United States?

Washington D.C.

```

# Acknowledgments

This project is greatly influenced by the following projects:

- [GPT-2] (https://github.com/openai/gpt-2)
- [GPT-3] (https://github.com/openai/gpt-3)
- [Learning to Summarize from Human Feedback] (https://github.com/openai/summarize-from-feedback)
- [nanoGPT] (https://github.com/karpathy/nanoGPT)
