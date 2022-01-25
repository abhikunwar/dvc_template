import pandas as pd
import numpy as np
import argparse
import logging

from sklearn import metrics
from src.utils import common
import os
import joblib
import json

STAGE = "four"
def evaluate(config_path,params_path):
    config_yaml = common.read_yaml(config_path)
    param_yaml = common.read_yaml(params_path)

    vec_dir_path = os.path.join(config_yaml['artifacts']['ARTIFACTS_DIR'],config_yaml['artifacts']['featurize_dir'])
    vec_test_data_name = config_yaml['artifacts']['featurize_test_data']
    vec_test_data_path = os.path.join(vec_dir_path,vec_test_data_name)
    vec_test_data = joblib.load(vec_test_data_path)

    model_dir_path = os.path.join(config_yaml['artifacts']['ARTIFACTS_DIR'],config_yaml['artifacts']['model_dir'])
    model_path = os.path.join(model_dir_path,config_yaml['artifacts']['model_file'])
    model = joblib.load(model_path)

    y_test = np.squeeze(vec_test_data[:,1].toarray())
    X_test = vec_test_data[:,2:]

    prediction_by_class = model.predict_proba(X_test)
    # print(prediction_by_class)
    prediction = [1 if i > .5 else 0 for i in prediction_by_class[:,1]]

    # print(prediction)
    from sklearn.metrics import classification_report
    report = classification_report(y_test,prediction)
    # print(report)
    # logging.log(f"the final model metrics is: {report} ")

    predict = prediction_by_class[:,1]
    # print(predict)

    # saving the tpr,fpr,threshold into json file
    prc_file_name = config_yaml['metrix']['prc']
    auc_roc_file_name = config_yaml['metrix']['roc']

    score_file_name =  config_yaml['scores']['SCORE']

    # print(y_test)
    y_test = np.array(y_test)
    # print(type(y_test))
    # # print("***************")
    # print(predict)
    # print(type(predict))
    # print(prc_file_name,auc_roc_file_name)
    precision,recall,p_threshold = metrics.precision_recall_curve(y_test,predict)
    fpr,tpr,fpr_tpr_threshold = metrics.roc_curve(y_test,predict)

    # i am going to put precision recall and threshold in a json file
    #this json file will be the output for step4
    #dvc studio has some issue to raead the json output in a folder
    # so will keep json output on the main folder

    scores_p_r = {"precision":precision[:-1].tolist() ,"recall":recall[:-1].tolist(),"p_threshold":p_threshold.tolist()}
    scores_tpr_fpr = {"fpr":fpr[:-1],"tpr":tpr[:-1],"fpr_tpr_threshold":fpr_tpr_threshold}
    # print(scores_p_r)
    # print(len(scores_p_r['precision']))
    # print(len(scores_p_r['recall']))
    # print(len(scores_p_r['p_threshold']))

    # print(list(zip(precision,recall,p_threshold)))
    df = pd.DataFrame(scores_p_r)

    # print(df.head())

    print(scores_p_r)

    with open(score_file_name,"w") as f:
        json.dump(scores_p_r,f,indent=4)
    
    # print(precision,recall,p_threshold)
    # print("***************")
    # print(fpr,tpr,fpr_tpr_threshold)

if __name__=="__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--config", "-c", default="configs/config.yaml")
    args.add_argument("--params", "-p", default="params.yaml")
    parsed_args = args.parse_args()

    

    try:
        logging.info("\n********************")
        logging.info(f">>>>> stage {STAGE} started <<<<<")
        evaluate(config_path=parsed_args.config, params_path=parsed_args.params)
        logging.info(f">>>>> stage {STAGE} completed!<<<<<\n")
    except Exception as e:
        logging.exception(e)
        raise e