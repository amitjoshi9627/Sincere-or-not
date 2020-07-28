import transformers
import os

DIR_ROOT = os.getcwd()
TRAIN_DATA_PATH = os.path.join(DIR_ROOT,"data/train.csv")
TEST_DATA_PATH = os.path.join(DIR_ROOT,"data/test.csv")
SUBMISSION_PATH = os.path.join("data/sample_submission.csv")

BERT_PATH = os.path.join(DIR_ROOT,"Bert_base_uncased")

BERT_DOWNLOAD_PATH = 'bert-base-uncased'    

TOKENIZER = transformers.BertTokenizer.from_pretrained(BERT_DOWNLOAD_PATH)
MODEL_PATH = os.path.join(DIR_ROOT,"Saved_Model/svm_model.pkl")

MAX_LEN = 64
BATCH_SIZE = 2