# CSE 5525 Final Project

This is our repo for the CSE 5525 final project. Our goal was to explore research I had done for another project.
In this extension we explored 2 GPT-2 models, trained on both orthography and phonetics.

# Dataset

We needed a dataset for doing edge probing, so we made one. It basically maps phonemes to lists of phonological features.

The format is:
```json
{
  "mappings": {
    "a": [0, 4, 5, -6, 7, ...]
  },
  "features": {
    "syllabic": 0,
    "stress": 1,
    ...
    "tense": n
  }
}
```

Where each phoneme `"a"` in this case, maps to a list of present phonological features, using negative numbers to mean "don't care".