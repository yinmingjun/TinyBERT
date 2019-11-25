TinyBERT
======== 
TinyBERT is 7.5x smaller and 9.4x faster on inference than BERT-base and achieves competitive performances in the tasks of natural language understanding. It performs a novel transformer distillation at both the pre-training and task-specific learning stages. The overview of TinyBERT learning is illustrated as follows: 

![image](tinybert_overview.png)

For more details about the techniques of TinyBERT, refer to our paper.

Release Notes
=============
First version: 2019/11/24

General Distillation
====================
In general distillation, we use the original BERT-base without fine-tuning as the teacher and a large-scale text corpus as the learning data. By performing the Transformer distillation on the text from general domain, we obtain a general TinyBERT which provides a good initialization for the task-specific distillation. 

General distillation has two steps: (1) generate the corpus of json format; (2) run the transformer distillation;

Step 1: Use `pregenerate_training_data.py` to produce the corpus of json format
```
 # ${BERT_BASE}$: this directory includes the BERT-base teacher model.
 
python pregenerate_training_data.py --train_corpus ${CORPUS_RAW} \ 
                  --bert_model ${BERT_BASE}$ \
                  --reduce_memory --do_lower_case \
                  --output_dir ${CORPUS_JSON}$                             
```

Step 2: Use `general_distill.py` to run the general distillation
```
 # ${STUDENT_CONFIG_DIR}$: this directory includes the config file of student_model.
 
python general_distill.py --pregenerated_data ${CORPUS_JSON}$ \ 
                          --teacher_model ${BERT_BASE}$ \
                          --student_model ${STUDENT_CONFIG_DIR}$ \
                          --reduce_memory --do_lower_case \
                          --train_batch_size 256 \
                          --output_dir ${GENERAL_TINYBERT_DIR}$
```


We also provide the models of general TinyBERT here and users can skip the general distillation.

[TinyBERT]()

[6-layer TinyBERT]()


Data Augmentation
=================
Data augmentation expands the task-specific training set. Learning more task-related examples, the generalization capabilities of student model can be further improved. We combine a pre-trained language model BERT and GloVe embeddings to do word-level replacement for data augmentation.

Use `data_augmentation.py` to run data augmentation.

```
python data_augmentation.py --pretrained_bert_model ${BERT_BASE_DIR}$ \
                            --glove_embs ${GLOVE_EMB}$ \
                            --glue_dir ${GLUE_DIR}$ \  
                            --task_name ${TASK_NAME}$
```


Task-specific Distillation
==========================
In the task-specific distillation, we re-perform the proposed Transformer distillation to further improve TinyBERT by focusing on learning the task-specific knowledge. 
The fine-tuned BERT-base model is acted as the teacher.

Task-specific distillation includes two steps: (1) intermediate layer distillation; (2) prediction layer distillation.

Step 1: use `task_distill.py` to run the intermediate layer distillation.
```
# ${FT_BERT_BASE_DIR}$: this directory contains the fine-tuned BERT-base model.

python task_distill.py --teacher_model ${FT_BERT_BASE_DIR}$ \
                       --student_model ${GENERAL_TINYBERT_DIR}$ \
                       --data_dir ${TASK_DIR}$ \
                       --task_name ${TASK_NAME}$ \ 
                       --output_dir ${TMP_TINYBERT_DIR}$ \
                       --aug_train \
                       --do_lower_case 
```


Step 2: use `task_distill.py` to run the prediction layer distillation.

```
python task_distill.py --pred_distill  \
                       --teacher_model ${FT_BERT_BASE_DIR}$ \
                       --student_model ${TMP_TINYBERT_DIR}$ \
                       --data_dir ${TASK_DIR}$ \
                       --task_name ${TASK_NAME}$ \
                       --output_dir ${TINYBERT_DIR}$ \
                       --aug_train  \  
                       --do_lower_case  
```


We here provide the fine-tuned TinyBERTs for all tasks in GLUE.

[MNLI Fine-tuned TinyBERT]()

[QQP Fine-tuned TinyBERT]()

[SST-2 Fine-tuned TinyBERT]()

[QNLI Fine-tuned TinyBERT]()

[MRPC Fine-tuned TinyBERT]()

[RTE Fine-tuned TinyBERT]()

[CoLA Fine-tuned TinyBERT]()


Evaluation
==========================
The `task_distill.py` also provides the evaluation by running the following command:

```
python task_distill.py --do_eval \
                       --teacher_model ${FT_BERT_BASE_DIR}$ \
                       --student_model ${TINYBERT_DIR}$ \
                       --data_dir ${TASK_DIR}$ \
                       --task_name ${TASK_NAME}$ \
                       --output_dir ${OUTPUT_DIR}$ \
                       --do_lower_case
```

To Dos
=========================
* Evaluate TinyBERT on Chinese tasks.
* Tiny*: Use other pre-trained language models as the teacher in TinyBERT learning.
* Release better general TinyBERTs.
