"""Shared utilities for finetuning scripts (SFT, DPO, RM).

Includes:
- parse_conversation: parse Human/Assistant transcript into messages
- ensure_last_turn_is_user: ensure the prompt ends on a 'user' turn
- apply_chat_template_list: render lists of conversations via tokenizer chat template
- load_df: lightweight dataframe loader with extension-based dispatch
- load_chat_tokenizer: tokenizer loader with chat-friendly defaults
"""

from typing import Dict, List
import re
import pandas as pd
from transformers import AutoTokenizer
from datasets import Dataset
import json

def parse_conversation(prompt_text: str) -> List[Dict[str, str]]:
    """Parse a multi-turn transcript into a messages list.

    Expected format in `prompt_text`:
        "Human: ...\n\nAssistant: ...\n\nHuman: ...\n\n..."
    """
    messages: List[Dict[str, str]] = []
    parts = re.split(r'(Human:|Assistant:)', prompt_text)

    current_role = None
    current_content = ""

    for part in parts:
        if part == "Human:":
            if current_role and current_content.strip():
                messages.append({"role": current_role, "content": current_content.strip()})
            current_role = "user"
            current_content = ""
        elif part == "Assistant:":
            if current_role and current_content.strip():
                messages.append({"role": current_role, "content": current_content.strip()})
            current_role = "assistant"
            current_content = ""
        else:
            current_content += part

    if current_role and current_content.strip():
        messages.append({"role": current_role, "content": current_content.strip()})

    return messages


def ensure_last_turn_is_user(messages: List[Dict[str, str]], fallback_prompt_text: str) -> List[Dict[str, str]]:
    """Ensure the conversation ends with a 'user' turn.

    If it ends with an 'assistant' turn, trim to the last 'user' turn if possible;
    otherwise, collapse to a single 'user' message using the provided fallback text.
    """
    if not messages:
        return [{"role": "user", "content": fallback_prompt_text}]

    if messages[-1]["role"] == "user":
        return messages
    raise ValueError("Conversation must end with a 'user' turn")


def apply_chat_template_list(tokenizer, conversations: List[List[Dict[str, str]]]) -> List[str]:
    """Apply tokenizer chat template to a list of conversations and return rendered texts."""
    return [tokenizer.apply_chat_template(conv, tokenize=False) for conv in conversations]


def load_df(path: str) -> pd.DataFrame:
    """Load a dataframe from csv/json/parquet based on file extension.

    Raises ValueError for unsupported extensions.
    """
    if path.endswith(".parquet"):
        return pd.read_parquet(path)
    if path.endswith(".csv"):
        return pd.read_csv(path)
    if path.endswith(".json"):
        return pd.read_json(path, orient="records", lines=True)
    raise ValueError("Data file must be .parquet, .csv, or .json")


def load_chat_tokenizer(model_or_path: str):
    """Load a tokenizer with chat-friendly defaults (pad to EOS, left padding)."""
    tokenizer = AutoTokenizer.from_pretrained(model_or_path, trust_remote_code=True)
    try:
        tokenizer.pad_token_id = tokenizer.eos_token_id
        print(f"Set pad_token_id to {tokenizer.decode(tokenizer.pad_token_id)}")
    except Exception:
        pass
    try:
        tokenizer.padding_side = "left"
        tokenizer.truncation_side = "left"
    except Exception:
        pass
    return tokenizer


def prepare_reward_dataset(df: pd.DataFrame, tokenizer: AutoTokenizer, model_name: str) -> Dataset:
    """Build dataset with 'chosen'/'rejected' as full conversations (list of messages).

    Starts from a dataframe with 'prompt', 'chosen', 'rejected' columns.
    'prompt' is in the format of "Human: ...\n\nAssistant: ...\n\nHuman: ..."
    'chosen' and 'rejected' are the responses to the prompt.

    We first parse the prompt into a messages list, then construct the chosen/rejected conversations.

    Each item contains:
      - chosen: messages + final assistant with chosen text
      - rejected: messages + final assistant with rejected text
      - length: token length used for bucketing when group_by_length=True
    """
    if not ("chosen" in df.columns and "rejected" in df.columns):
        raise ValueError("Dataset must contain 'chosen' and 'rejected' columns")

    records = []
    for _, row in df.iterrows():
        prompt_text = row.get("prompt", "")
        messages = parse_conversation(prompt_text)
        messages = ensure_last_turn_is_user(messages, prompt_text)

        chosen_text = (row.get("chosen") or "").strip()
        rejected_text = (row.get("rejected") or "").strip()

        chosen_conv = messages + [{"role": "assistant", "content": chosen_text}]
        rejected_conv = messages + [{"role": "assistant", "content": rejected_text}]

        # Compute token-lengths for bucketing. Use the max of (chosen, rejected) lengths.
        chosen_text = tokenizer.apply_chat_template(chosen_conv, tokenize=False)
        rejected_text = tokenizer.apply_chat_template(rejected_conv, tokenize=False)

        chosen_ids = tokenizer.encode(chosen_text, add_special_tokens=False)
        rejected_ids = tokenizer.encode(rejected_text, add_special_tokens=False)
        example_length = max(len(chosen_ids), len(rejected_ids))

        records.append({
            "chosen": chosen_conv,
            "rejected": rejected_conv,
            "length": int(example_length),
        })

    return Dataset.from_list(records)
