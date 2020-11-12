export MAX_LENGTH=128
export BERT_MODEL=bert-base-multilingual-cased
python3 scripts/preprocess.py train.txt.tmp $BERT_MODEL $MAX_LENGTH > train.txt
python3 scripts/preprocess.py dev.txt.tmp $BERT_MODEL $MAX_LENGTH > dev.txt
python3 scripts/preprocess.py test.txt.tmp $BERT_MODEL $MAX_LENGTH > test.txt
cat train.txt dev.txt test.txt | cut -d " " -f 2 | grep -v "^$"| sort | uniq > labels.txt
export OUTPUT_DIR=germeval-model
export BATCH_SIZE=32
export NUM_EPOCHS=3
export SAVE_STEPS=750
export SEED=1

python3 run_ner.py \
--task_type NER \
--data_dir data/ \
--labels ./labels.txt \
--model_name_or_path "bertcrf" \
--config_name "bert-base-uncased" \
--tokenizer_name "bert-base-uncased" \
--output_dir $OUTPUT_DIR \
--max_seq_length  $MAX_LENGTH \
--num_train_epochs $NUM_EPOCHS \
--per_gpu_train_batch_size $BATCH_SIZE \
--save_steps $SAVE_STEPS \
--seed $SEED \
--do_train \
--task_name "twitter2017" \
--feature_type "Object" \
--fine_tune_cnn False \
--learning_rate 5e-5 \
--bert_lr 5e-5 \
--classifier_lr  5e-5 \
--crf_lr  1e-3 \

