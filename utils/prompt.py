"""Build prompt-completion pairs using predefined format"""
from typing import Tuple, List, Mapping, Text


END_TOKEN = ' END'


def build_prompt_completion(prompt: str, completion: str, context=None) -> Tuple[str]:
    """Build prompt and completion pair using pre-defined format.

    if context is none:
    "\n\n### User: <prompt>\n\n### Assistant:", " <completion> END"

    otherwise:
    "\n\n### Context: <context>\n\n### User: <prompt>\n\n### Assistant:", " <completion> END"

    """
    assert prompt is not None and len(prompt) > 0
    assert completion is not None and len(completion) > 0

    formatted_prompt = ""
    if context is not None and len(context) > 0:
        formatted_prompt = f"\n\n### Context: {context.strip()}"

    formatted_prompt += f"\n\n### User: {prompt.strip()}\n\n### Assistant:"

    formatted_completion = " " + completion.strip() + END_TOKEN

    return formatted_prompt, formatted_completion


def build_conversation_prompt_completions(prompt_completion_pairs: List[Mapping[Text, Text]], context=None) -> Tuple[str]:
    """Build conversational style prompt and completion pairs using pre-defined format.

    if context is none:
    "\n\n### User: <prompt_1>\n\n### Assistant: <completion_1>\n\n### User: <prompt_2>### Assistant:", " <completion_2> END"

    otherwise:
    "\n\n### Context: <context>\n\n### User: <prompt_1>\n\n### Assistant: <completion_1>\n\n### User: <prompt_2>### Assistant:", " <completion_2> END"

    """
    assert prompt_completion_pairs is not None and len(prompt_completion_pairs) > 0

    formatted_prompt = ""

    if context is not None and len(context) > 0:
        formatted_prompt = f"\n\n### Context: {context.strip()}"

    for i, pair in enumerate(prompt_completion_pairs):
        prompt = pair['prompt']
        completion = pair['completion']

        formatted_prompt += f"\n\n### User: {prompt.strip()}"

        if i < len(prompt_completion_pairs) - 1:
            formatted_prompt += f"\n\n### Assistant: {completion.strip()}"

    formatted_prompt += "\n\n### Assistant:"

    formatted_completion = " " + prompt_completion_pairs[-1]['completion'].strip() + END_TOKEN

    return formatted_prompt, formatted_completion
