# miniGPT
Try to implement a minimum version of GPT-2 model for research and education purpose. Although we focus on the GPT-2 model, the procedure is suitable to training any autoregressive language models.


**Note**:
Our initial goal was trying to build a full pipeline including RLHF module following the InstructGPT paper, however we soon realized that the pre-trained GPT-2 model is inadequate for complex tasks like answering open-domain questions due to it's small model size. Instead, we will try to implement that pipeline in another project using LLaMA.


## What we got
* data-preprocessing, pre-training, fine-tuning scripts
* support PyTorch FSDP for distributed training
* support fine-tunning from loading OpenAI pre-trained weights (through hugging face transformers)
* support fine-tunning with LoRA (no FSDP support since PyTorch 2.0.1 has some bug)


# Environment and Requirements
* Python        3.10.6
* PyTorch       2.0.1
* Tensorboard   2.13.0


# Code Structure
*   `configs` directory contains all the training configurations like model type, data source, number of iterations, learning rate etc.
*   `models` directory contains GPT-2 model definition.
*   `utils` directory contains helper modules like checkpoint, logging, tokenization etc.
*   `prepare_data_pretrain.py` contains code to clean up raw text before build pre-training datasets.
*   `build_pretrain_datasets.py` contains code to build pre-train datasets (tokenize, and them save the dataset in Numpy memmap structure).
*   `prepare_data_finetune.py` contains code to process raw data before build fine-tunning datasets.
*   `build_finetune_datasets.py` contains code to build fine-tunning datasets (tokenize, and them save the dataset to .jsonl files).
*   `custom_dataset.py` contains code for custom dataset instances for pre-training and fine-tunning.
*   `pretrain.py` contains code to run pre-training using FSDP.
*   `finetune.py` contains code to run full fine-tunning using FSDP.
*   `finetune_lora.py` contains code to run LoRA fine-tunning.
*   `generate_pretrained.py` contains code to evaluate the pre-trained model (predict next token).
*   `generate_finetuned.py` contains code to evaluate the fine-tuned model (answer questions).
*   `convert_hf_checkpoint.py` contains code to convert OpenAI pre-trained GPT-2 weights to support our model, so we can load it to start fine-tunning.
*   `convert_lora_checkpoint.py` contains code to convert fine-tunned LoRA weights to a full state_dict checkpoint.


# Project Setup

```
python3 -m pip install --upgrade pip setuptools

python3 -m pip install -r requirements.txt 
```


# Download and prepare datasets
You need download the source files for the individual dataset from the Internet, then using our data preparation and build dataset scripts to turn them into ready to use datasets. We don't provide any ready to use dataset files. Our simple yet not perfect data preparation scripts should be able to handle most common datasets formats (.txt, .json, .jsonl files) with minimum changes.

Once you have downloaded the source files, we can follow these two process to build the training datasets:
* stage 1: raw text files (.txt, .json, .json files) are pre-processed and cleaned up using `prepare_data_pretrain.py` and `prepare_data_finetune.py` scripts.
* stage 2: cleaned up text is then turned into token ids using the `build_pretrain_datasets.py` and `build_finetune_datasets.py` scripts.


## Wikipedia datasets

We can use wikiextractor to extract the rax text from Wikipedia dumps, so later we can use the script to prepare and clean up the text.

```
wikiextractor  ~/Downloads/enwiki-latest-pages-articles.xml.bz2 -b 200M --no-templates --processes 24 --json -o ./raw_data/enwiki_latest

# add .json file extension
find ./raw_data/enwiki_latest -type f -name 'wiki*' -exec sh -c 'x="{}"; mv "$x" "${x}.jsonl"' \;

```


# Pre-training

This examples shows how to lunch the pre-training script on a machine with 4 GPUs.
```
torchrun --standalone --nproc_per_node 4 pretrain.py
```


# Fine-tunning

## Full fine-tunning
This examples shows how to lunch the full fine-tunning script on a machine with 4 GPUs.
```
torchrun --standalone --nproc_per_node 4 finetune.py
```


## LoRA fine-tunning

As of PyTorch 2.0.1, the FSDP module does not support LoRA, so we have commented out the code related to FSDP inside `finetune_lora.py`. This means the code does not support running on multiple GPUs at the moment. We hope with new PyTorch release, the issue will be fixed soon.

This examples shows how to lunch the LoRA fine-tunning script on a machine with 1 GPUs.
```
torchrun --standalone --nproc_per_node 1 finetune_lora.py
```


# Generate

We have provided some sample tasks in the `generate_pretrained.py` and `generate_finetuned.py` to get you started. Note that the pre-trained model only predicts the next token, so the response is really not great if the input is some general question.

We can clearly see that the fine-tuned model can answer some (but not all) questions in a reasonably manner. Model size plays a pretty important role here, as we observed, for general question answering tasks, we should use at least the 774M parameters version `gpt2-large`, or using the biggest one `gpt2-xl`.

```
python generate_pretrained.py


python generate_finetuned.py
```


# Acknowledgments

This project is greatly influenced by the following projects:
* [GPT-2] (https://github.com/openai/gpt-2)
* [GPT-3] (https://github.com/openai/gpt-3)
* [Learning to Summarize from Human Feedback] (https://github.com/openai/summarize-from-feedback)
* [nanoGPT] (https://github.com/karpathy/nanoGPT)


The following projects have been very helpful to us to implement the LoRA fine-tunning scripts:
* [Lit-LLaMA] (https://github.com/Lightning-AI/lit-llama)
* [LoRA] (https://github.com/microsoft/LoRA)




