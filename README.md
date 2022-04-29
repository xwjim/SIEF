# Document-Level Relation Extraction with Sentences Importance Estimation and Focusing
PyTorch implementation for NAACL 2022 Main Conference paper: [Document-Level Relation Extraction with Sentences Importance Estimation and Focusing](https://arxiv.org/abs/2204.12679).

> Document-level relation extraction (DocRE) aims to determine the relation between two entities from a document of multiple sentences.
Recent studies typically represent the entire document by sequence- or graph-based models to predict the relations of all entity pairs. However, we find that such a model is not robust and exhibits bizarre behaviors: it predicts correctly when an entire test document is fed as input, but errs when non-evidence sentences are removed. To this end, we propose a Sentence Importance Estimation and Focusing (SIEF) framework for DocRE, where we design a sentence importance score and a sentence focusing loss, encouraging DocRE models to focus on evidence sentences. Experimental results on two domains show that our SIEF not only improves overall performance, but also makes DocRE models more robust. Moreover, SIEF is a general framework, shown to be effective when combined with a variety of base DocRE models.

## 0. Package Description
```
SIEF/
├─ code/
    ├── checkpoint/: save model checkpoints
    ├── logs/: save training / evaluation logs
    ├── models/: base models
        ├── SIEF.py: sief modules 
    ├── config.py: process command arguments
    ├── data.py: define Datasets / Dataloader
    ├── test.py: evaluation code
    ├── train.py: training code
    ├── utils.py: some tools for training / evaluation
    ├── *.sh: training / evaluation shell scripts
├─ data/docred: raw data and preprocessed data about DocRED dataset
    ├── prepro_data/
├─ LICENSE
├─ README.md
```

## 1. Environments

- python         (3.7.9)
- cuda           (11.4)
- Pop!_OS 20.04  (5.11.0-7620-generic)

## 2. Dependencies

- numpy          (1.20.3)
- matplotlib     (3.4.2)
- torch          (1.10.0)
- transformers   (4.17.0)
- scikit-learn   (0.24.2)
- wandb          (0.12.14)

## 3. Preparation

### 3.1. Dataset
- Download data from [Google Drive link](https://drive.google.com/drive/folders/1c5-0YwnoJx8NS6CV2f-NoTHR__BdkNqw) shared by DocRED authors

- Put `train_annotated.json`, `dev.json`, `test.json`, `word2id.json`, `ner2id.json`, `rel2id.json`, `vec.npy` into the directory `data/docred/`

- If you want to use other datasets, please first process them to fit the same format as DocRED.

### 3.2. Pre-trained Language Models
The package *transformers* would take some time to download the pretrained model for the first time.

## 4. Training
You can train the HeterGSAN with SIEF as following. It is noted that about twice GPU memory is needed training with SIEF. 

```bash
>> cd code
>> ./runXXX.sh gpu_id   # like ./run_BERT.sh 2
>> tail -f -n 2000 logs/train_xxx.log
```

## 5. Evaluation

```bash
>> cd code
>> ./evalXXX.sh gpu_id threshold(optional)  # like ./eval_BERT.sh 0 0.5521
>> tail -f -n 2000 logs/test_xxx.log
```

PS: we recommend to use threshold = -1 (which is the default, you can omit this arguments at this time) for dev set, 
the log will print the optimal threshold in dev set, and you can use this optimal value as threshold to evaluate test set.

## 6. Submission to LeadBoard (CodaLab)
- You will get json output file for test set at step 5. 

- And then you can rename it as `result.json` and compress it as `result.zip`. 

- At last,  you can submit the `result.zip` to [CodaLab](https://competitions.codalab.org/competitions/20717#participate-submit_results).

## 7. License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 8. Citation

Our code is adapted from [GAIN](https://github.com/DreamInvoker/GAIN). Thanks for their good work. If you use this work or code, please kindly cite the following paper:

```bib
@inproceedings{wangx-NAACL-2022-sief,
 author = {Wang Xu, Kehai Chen, Lili Mou and Tiejun Zhao},
 booktitle = {The Conference of the North American Chapter of the Association for Computational Linguistics},
 title = {Document-Level Relation Extraction with Sentences Importance Estimation and Focusing},
 year = {2022}
}
```

