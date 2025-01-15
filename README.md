# A Proposal of Post-OCR Spelling Correction Using Monolingual Byte-level Language Models

### Abstract
This work presents a proposal for a spelling corrector using monolingual byte-level language models (Monobyte) for the post-OCR task in texts produced by Handwritten Text Recognition (HTR) systems. We evaluate three Monobyte models, based on Google’s ByT5, trained separately on English, French, and Brazilian Portuguese. The experiments evaluated three datasets with 21st century manuscripts: IAM, RIMES, and BRESSAY. In the IAM, Monobyte achieves reductions of 2.24\% in character error rate (CER) and 26.37\% in word error rate (WER). In RIMES, reductions are 13.48\% (CER) and 33.34\% (WER), while in BRESSAY, Monobyte improves CER by 12.78\% and WER by 40.62\%. The BRESSAY results surpass results reported in previous works using a multilingual ByT5 model. Our findings demonstrate the effectiveness of byte-level tokenization in noisy text and underscore the potential of computationally efficient, monolingual models.

---

*Fine-tuned models*: https://huggingface.co/savi8sant8s/monobyte-spelling-corrector

### Instructions to run experiments

#### 1 - Create Conda environment:
```sh
conda create --name=monobyte_spelling_corrector python=3.10 -y
conda activate monobyte_spelling_corrector
pip install -r requirements.txt
```

#### 2 - Fine-tuning:
```sh
python3 fine_tuning.py \
  --model monobyte/byt5-mono-pt-v1 \
  --dataset bressay \
  --output_folder models/pt
```

#### 3 - Inference:
```sh
python3 inference.py \
    --model models/pt \
    --ocr_predictions bluche flor puigcerver \
    --dataset bressay \
    --output_folder corrections/bressay
```

#### 4 - Calculate metrics:
```sh
python3 metrics.py \
    --dataset bressay \
    --ocr_predictions bluche flor puigcerver \
    --output_corrections corrections/bressay
```

### Citation
```bibtex
@inproceedings{araujo2025proposal,
  author    = {Sávio Santos de Araújo and Byron Leite Dantas Bezerra and Arthur Flor de Souza Neto},
  title     = {A Proposal of Post-OCR Spelling Correction Using Monolingual Byte-level Language Models},
  booktitle = {Proceedings of the ACM Symposium on Document Engineering 2025 (DocEng '25)},
  year      = {2025},
  publisher = {ACM},
  doi       = {10.1145/3704268.3748673}
}
