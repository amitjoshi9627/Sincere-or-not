import config
from model import BertForFeatureExtraction
import engine
import utils
import pandas as pd

def _save_submission(results):
    ind=260000
    submission = pd.read_csv(config.SUBMISSION_PATH)
    submission.iloc[ind:ind+10000,1] = results
    file_name = config.SUBMISSION_PATH
    utils.pandas_to_csv(submission,file_name)

def _get_test_results(Model):
    clf = utils.load_model(config.MODEL_PATH)
    test_features = engine.get_features(Model,train=False)
    test_predictions = engine.test_results(clf,test_features)
    _save_submission(test_predictions)

def _training(Model):
    features, targets = engine.get_features(Model,train=True)
    X_train,X_test,y_train,y_test = utils.train_test_split(features, targets, test_size=0.3)
    classifier = engine.train_fn(X_train,y_train)
    utils.save_model(classifier,config.MODEL_PATH)
    predictions = engine.eval_fn(classifier,X_test)
    accuracy = utils.accuracy_score(predictions,y_test)
    print("Accuracy Score:",accuracy)

def run():
    Model = BertForFeatureExtraction()
    # _training(Model)
    _get_test_results(Model)

    

if __name__ == "__main__":
    run()