# Revealing Personality Traits: A New Benchmark Dataset for Explainable Personality Recognition on Dialogues

This is the repository of our EMNLP 2024 Main conference paper ["Revealing Personality Traits: A New Benchmark Dataset for Explainable Personality Recognition on Dialogues"](<https://aclanthology.org/2024.emnlp-main.1115/>).

## PersonalityEvd Dataset

Our PersonalityEvd dataset is organized under the `Dataset` folder \
1. dialogue.json: contains dialogue data for 72 characters, totaling 1924 dialogues
2. EPR-State Task folder: annotation for EPR-State task
- train_annotation.json：annotation of train set, 51 characters
- valid_annotation.json：annotation of valid set, 7 characters
- test_annotation.json：annotation of test set, 14 characters
  - file format:
    ```Python
    "character": {
        "dlg_num": ...,
        "annotation": {
            "dialogue id": {"openness": {"level": ... ,"utt_id": ... ,"nat_lang": ...},  "conscientiousness": ... , },
      ...
     }}
    ```
    - "level"：personality state label, "high"、"low" or "unsure"
    - "utt_id"：evidence utterance ids
    - "nat_lang"：natural language evidence composed of `utterance summaries` and `personality characteristics`
1. EPR-Trait Task folder: annotation for EPR-Trait task. Due to the limited amount of data, 3-fold cross validation was adopted.
- 3_folds.json: contain the characters of each fold
- trait_annotation.json:
  - file format:
    ```Python
    "character": {
          "openness": {"level": ... , "dlg_id": ... , "nat_lang": ... },
          "conscientiousness": {"level": ... , "dlg_id": ... , "nat_lang": ... },
          ...
      }
    ```
    - "level"：personality trait label, "high"、"low" or "unsure"
    - "dlg_id" evidence dialogue ids, using "#" to distinguish the IDs of three facets of the target BF dimension, ";" to distinguish different performence.
    - "nat_lang"：natural language evidence composed of `dialogue summaries` and `personality characteristics`

## Model
There are bash scripts in folder `Code of EPR-State Task/ChatGLM/sh` to trian or test the model.

We use the code from this open-source repository [ChatGLM-Finetuning](<https://github.com/liucongg/ChatGLM-Finetuning>), and we are very grateful to the author.

## Cite
If you use our codes or your research is related to our work, please kindly cite our paper:
```
@inproceedings{sun-etal-2024-revealing,
    title = "Revealing Personality Traits: A New Benchmark Dataset for Explainable Personality Recognition on Dialogues",
    author = "Sun, Lei  and
      Zhao, Jinming  and
      Jin, Qin",
    editor = "Al-Onaizan, Yaser  and
      Bansal, Mohit  and
      Chen, Yun-Nung",
    booktitle = "Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing",
    month = nov,
    year = "2024",
}
```
Please contact leisun@ruc.edu.cn once you have any problems.
