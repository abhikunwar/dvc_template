import pandas as pd
import argparse
import logging
from src.utils import common
from src.utils import models
import os
import joblib
import numpy as np
STAGE = "Three"


def train(config_path,params_path):
    yaml_file = common.read_yaml(config_path)
    param_file = common.read_yaml(params_path)
    print(param_file)

    vec_dir_path = os.path.join(yaml_file['artifacts']['ARTIFACTS_DIR'],yaml_file['artifacts']['featurize_dir'])
    vec_train_data_name = yaml_file['artifacts']['featurize_train_data']
    vec_train_data_path = os.path.join(vec_dir_path,vec_train_data_name)
    vec_train_data = joblib.load(vec_train_data_path)
    # model path and directory
    model_dir = os.path.join(yaml_file['artifacts']['ARTIFACTS_DIR'],yaml_file['artifacts']['model_dir'])
    print("model_dir",model_dir)
    # create model dir
    common.create_directories([model_dir])
    print("model dir create")
    # model path
    model_path = os.path.join(model_dir,yaml_file['artifacts']['model_file'])
    # print(vec_train_data.shape)
    y_train = np.squeeze(vec_train_data[:,1].toarray())
    X_train = vec_train_data[:,2:]
    logging.info(f"input matrix size : {vec_train_data.shape}")
    logging.info(f"X matrix size : {X_train.shape}")
    logging.info(f"input label size : {y_train.shape}")

    n_est = param_file['train']['n_est']
    min_sample_split = param_file['train']['min_sample_split']
    n_jobs = param_file['train']['n_jobs']
    ram_st = param_file['train']['random_st']

    model = models.train_model(X_train,y_train,n_est,min_sample_split,n_jobs,ram_st)
    joblib.dump(model,model_path)


    

    

if __name__=="__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--config", "-c", default="configs/config.yaml")
    args.add_argument("--params", "-p", default="params.yaml")
    parsed_args = args.parse_args()

    

    try:
        logging.info("\n********************")
        logging.info(f">>>>> stage {STAGE} started <<<<<")
        train(config_path=parsed_args.config, params_path=parsed_args.params)
        logging.info(f">>>>> stage {STAGE} completed!<<<<<\n")
    except Exception as e:
        logging.exception(e)
        raise e