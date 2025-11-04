"""Aux Reward functions from openr1 for GRPO training."""

import json,math
import re
from typing import Dict


def cosine_scaled_reward(predict_str, is_correct,     
    min_value_wrong: float = -0.5,
    max_value_wrong: float = 0.0,
    min_value_correct: float = 0.0,
    max_value_correct: float = 0.0,
    max_len: int = 500,
    limited_len: int = 4096):
    """Reward function that scales based on completion length using a cosine schedule.

    Shorter correct solutions are rewarded more than longer ones.
    Longer incorrect solutions are penalized less than shorter ones.

    Args:
        predict_str: predict answer
        is_correct: correct or not

    This function is parameterized by the following arguments:
        min_value_wrong: Minimum reward for wrong answers
        max_value_wrong: Maximum reward for wrong answers
        min_value_correct: Minimum reward for correct answers
        max_value_correct: Maximum reward for correct answers
        max_len: Maximum length for scaling
    """
    gen_len = len(predict_str)

    if gen_len >= limited_len:
        return -0.5
    
    if gen_len >=max_len:
        return 0

    # Apply cosine scaling based on length
    progress = gen_len / max_len
    cosine = math.cos(progress * math.pi)

    if is_correct:
        min_value = min_value_correct
        max_value = max_value_correct
    else:
        # Swap min/max for incorrect answers
        min_value = max_value_wrong
        max_value = min_value_wrong


    reward = min_value + 0.5 * (max_value - min_value) * (1.0 + cosine)
    return reward

def repetition_penalty_reward(predict_str, ngram_size: int = 20, max_penalty: float = -0.5):
    """
    Computes N-gram repetition penalty as described in Appendix C.2 of https://arxiv.org/abs/2502.03373.
    Reference implementation from: https://github.com/eddycmu/demystify-long-cot/blob/release/openrlhf/openrlhf/reward/repetition.py

    Args:
    ngram_size: size of the n-grams
    max_penalty: Maximum (negative) penalty for wrong answers
    reward function the penalizes repetitions
    ref implementation: https://github.com/eddycmu/demystify-long-cot/blob/release/openrlhf/openrlhf/reward/repetition.py

    Args:
        completions: List of model completions
    """
    if max_penalty > 0:
        raise ValueError(f"max_penalty {max_penalty} should not be positive")

    def zipngram(text: str, ngram_size: int):
        words = text.lower().split()
        return zip(*[words[i:] for i in range(ngram_size)])

    completion = predict_str
    
    if completion == "":
        return 0.0
    if len(completion.split()) < ngram_size:
        return 0.0

    ngrams = set()
    total = 0
    for ng in zipngram(completion, ngram_size):
        ngrams.add(ng)
        total += 1

    scaling = 1 - len(ngrams) / total
    reward = scaling * max_penalty

    return reward
