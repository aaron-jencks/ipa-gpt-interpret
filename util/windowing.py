import random
from typing import Tuple, Set


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


def make_random_window(sentence: str, max_l: int, min_l: int, decay: float, phonemes: Set[str]) -> Tuple[int, int]:
    """
    find a random window selected from a sentence
    max and min are suggestions, the algorithm will ensure that the window starts and ends on a boundary,
    even if it has to go over the limit
    :param sentence:
    :param max_l: max window size
    :param min_l: min window size
    :param decay: controls the geometric decay
    :param phonemes: the phonemes set to use for boundary detection
    :return: the start and end position of the window
    """
    ws = find_random_window_size(len(sentence), max_l, min_l, decay)
    max_start = len(sentence) - ws

    start = random.randint(0, max_start)
    # start on a phoneme boundary
    while start > 0 and sentence[start] in phonemes:
        start -= 1
    start += 1

    end = start + ws
    # end on a phoneme boundary
    while end < len(sentence) and sentence[end] in phonemes:
        end += 1
    end -= 1

    return start, end
