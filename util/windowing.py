import random
from typing import Tuple


def find_random_window_size(sentence_len: int, max_l: int, min_l: int, decay: float) -> int:
    """
    randomly selects a window size, biasing towards smaller sizes
    :param sentence_len:
    :param max_l:
    :param min_l:
    :param decay: controls the geometric decay
    :return: the randomly selected window size
    """
    max_l = min(max_l, sentence_len)
    if max_l <= min_l:
        raise ValueError('max_l <= min_l')
    lengths = list(range(min_l, max_l + 1))
    weights = [decay ** (l - min_l) for l in lengths]
    window_size = random.choices(lengths, weights=weights, k=1)[0]
    return window_size


def make_random_window(sentence: str, max_l: int, min_l: int, decay: float) -> Tuple[int, int]:
    """
    find a random window selected from a sentence
    :param sentence:
    :param max_l: max window size
    :param min_l: min window size
    :param decay: controls the geometric decay
    :return: the start and end position of the window
    """
    ws = find_random_window_size(len(sentence), max_l, min_l, decay)
    max_start = len(sentence) - ws
    start = random.randint(0, max_start)
    end = start + ws
    return start, end
