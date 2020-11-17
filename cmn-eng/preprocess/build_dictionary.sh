#!/bin/bash
python tokenizer.py
# config the size of vocab=5000 and input english data en.txt
subword-nmt learn-bpe -s 5000 < en.txt > en_code.txt
# apply bpe,-c vocab <data.txt> result=en_refine.txt
subword-nmt apply-bpe -c en_code.txt < en.txt > en_refine.txt
# get vocab
subword-nmt get-vocab --input en_refine.txt --output en_vocab.txt

python build_dataset.py
