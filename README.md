# CSE 5525 Final Project

This is our repo for the CSE 5525 final project. Our goal was to explore research I had done for another project.
In this extension we explored 2 GPT-2 models, trained on both orthography and phonetics.

**Note:** You need git lfs to download the model files from the repo. You can learn how to install git lfs here: https://docs.github.com/en/repositories/working-with-files/managing-large-files/installing-git-large-file-storage

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

To generate this mapping file we use [mapping_generator.ipynb](mapping_generator.ipynb). 
It makes use of a reference spreadsheet obtained from the Hayes book on phonology. It also performs some basic analysis on the data.

# Inventory Analysis

The first thing we did was to explore the phonetic inventory of the languages to see how much overlap actually existed.
For this, the two [inventory_explorer](inventory_explorer.ipynb) files were used. They explore the character level inventory overlap.

For the token level analysis the [bpe-analysis.py](bpe-analysis.py) file was used. It tokenizes a dataset and then analyzes the token contents.
To replicate the results we collected you need to run the file with the following arguments:

**For IPA:**
```bash
python ./bpe-analysis.py \
  "data/tokenizers" "bpe-rus-pol-ipa-number-preservation-08-20-2025" \
  --dataset "iggy12345/pair_russian_polish_ipa" --feature "phonemes"
```

**For Orthographic:**
```bash
python ./bpe-analysis.py \
  "data/tokenizers" "bpe-rus-pol-normal-number-preservation-08-20-2025" \
  --dataset "iggy12345/pair_russian_polish_ipa" --feature "text"
```

In addition to these, there was also use of [dataset_statistics.ipynb](dataset_statistics.ipynb).
This supplies the feature distributions used in the paper.

# Hidden States

In order to train the linear probes efficiently we extracted the hidden states of the model separately,
that way we could load them from disk instead of running the model each time. To do this we used [hidden_state_extractor.py](hidden_state_extractor.py).
The model checkpoints you need for this along with the tokenizer files and the config files are supplied. For details see [config/default.json](config/default.json).

To extract the hidden states you simply run [hidden_state_extractor.py](hidden_state_extractor.py) as shown below:
```bash
python hidden_state_extractor.py
```
It's usage is:
```
usage: hidden_state_extractor.py [-h] [--default-config DEFAULT_CONFIG] [--cpus CPUS] [--debug] [--model-type MODEL_TYPE [MODEL_TYPE ...]] [--output-dir OUTPUT_DIR] [--batch-size BATCH_SIZE] [--accumulation-size ACCUMULATION_SIZE] config [config ...]

Extract per-token hidden states and save to numpy files

positional arguments:
  config                paths to config files

options:
  -h, --help            show this help message and exit
  --default-config DEFAULT_CONFIG
                        path to the default config file
  --cpus CPUS           number of cpus
  --debug               enable debug mode
  --model-type MODEL_TYPE [MODEL_TYPE ...]
                        The model type(s) to process
  --output-dir OUTPUT_DIR
                        Directory to save per-token hidden states
  --batch-size BATCH_SIZE
                        Batch size for hidden state extraction
  --accumulation-size ACCUMULATION_SIZE
                        The number of samples to store before saving
```

# Linear Probes

Now that the hidden states are acquired, the linear probes can be trained. To do this, the [probing-exp-preextracted.py](probing-exp-preextracted.py) is used.
Its usage is:
```
usage: probing-exp-preextracted.py [-h] [--default-config DEFAULT_CONFIG] [--cpus CPUS] [--debug] [--model-type MODEL_TYPE [MODEL_TYPE ...]] [--output-log OUTPUT_LOG] [--resume] [--average-span] [config ...]

positional arguments:
  config                paths to config files

options:
  -h, --help            show this help message and exit
  --default-config DEFAULT_CONFIG
                        path to the default config file
  --cpus CPUS           number of cpus
  --debug               enable debug mode
  --model-type MODEL_TYPE [MODEL_TYPE ...]
                        The model type
  --output-log OUTPUT_LOG
                        The file to store the final probing accuracies in
  --resume              Resume from latest checkpoint if available
  --average-span        Average all tokens in the span instead of using last token
```
To replicate our results it is necessary to use the `--average-span` flag. 
This indicates to average the hidden states of the entire span, as opposed to using just the last token's state.