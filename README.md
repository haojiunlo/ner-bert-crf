# NER-Bert-CRF
## Prepare training data
* Use [annotation tool](https://github.com/doccano/doccano) to annotate the data and export to JSONL(Text Label) format
* Convert to conll format.
```bash
python data_preprocess_jsonl_to_conll.py \
  --input_jsonl data/jsonl_format/data.json1 \
  --output_path data/conll_format/data.train
```

## Training
```bash
python task_sequence_labeling_ner_crf.py \
  --model_path pretrained_model/ \
  --config_file macbert_base_config.json \
  --vocab_file vocab.txt \
  --pretrained_weights pretrained_model/macbert_base_best_model_weights.h5 \
  --output_dir outputs/finetune \
  --data_dir data/conll_format \
  --train_file badcase_20210224_part1.train \
  --val_file badcase_20210224_part1.train
```
## Reference
[bert4keras](https://github.com/bojone/bert4keras)
[doccano](https://github.com/doccano/doccano)