from typing import List, Set, Dict


def identify_phonemes(s: str, phonemes: Set[str]) -> List[str]:
    """
    takes a string and returns the list of phonemes present in that string
    assumes that any character that is not a phoneme represents a word/phoneme boundary
    also assumes that any character that does represent a phoneme is indeed a phoneme
    also assumes that the first and last characters are at boundaries
    :param s: string to separate into phonemes
    :param phonemes: The set of valid phonemes
    :return: a list of strings representing phonemes
    """
    result = []
    current = ''
    for c in s:
        if c not in phonemes:
            if len(current) > 0:
                result.append(current)
            current = ''
            continue
        temp = current + c
        if temp not in phonemes:
            result.append(current)
            current = c
        else:
            current = temp
    if len(current) > 0:
        result.append(current)
    return result


def merge_features(a: Set[int], b: Set[int]) -> Set[int]:
    """
    merges two sets of features into a single set
    don't care scenarios are handles as follows (DC = Don't Care):

    | Presence in other set | Result |
    |        Y              |   Y    |
    |        N              |   DC   |
    |        DC             |   DC   |

    :param a:
    :param b:
    :return:
    """
    result = a & b
    # Handle don't care cases
    for phone_a in (a - result):
        if phone_a >= 0 or -phone_a not in b:
            result.add(phone_a)
    for phone_b in (b - result):
        if phone_b >= 0 or -phone_b not in a:
            result.add(phone_b)
    return result


def get_features(s: str, mapping: Dict[str, List[int]]) -> List[int]:
    """
    takes a string and returns a list of integers representing the features present in that string
    first determines which phonemes are present in that string
    then merges all of the phonological features of the phonemes into a single list
    assumes that the first and last characters are at boundaries
    :param s: string to find features for
    :param mapping: The phoneme mappings
    :return: a list of indices representing the features present in the string
    """
    result = set()
    phonemes = identify_phonemes(s, set(mapping.keys()))
    for phone in phonemes:
        result = merge_features(result, set(mapping[phone]))
    return list(result)
