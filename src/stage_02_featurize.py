import pandas as pd
import argparse
import logging
import argparse
import os
import shutil
from tqdm import tqdm
import logging
from src.utils.common import read_yaml, create_directories,get_df
import random
from sklearn.feature_extraction.text import CountVectorizer
from src.utils import featurize


STAGE = "STAGE_2" ## <<< change stage name 

logging.basicConfig(
    filename=os.path.join("logs", 'running_logs.log'), 
    level=logging.INFO, 
    format="[%(asctime)s: %(levelname)s: %(module)s]: %(message)s",
    filemode="a"
    )


def main(config_path, params_path):
    ## read config files
    config = read_yaml(config_path)
    params = read_yaml(params_path)
    art_dir = config['artifacts']['ARTIFACTS_DIR']
    featurize_dir = config['artifacts']['featurize_dir']
    featurize_train_data = config['artifacts']['featurize_train_data']
    featurize_test_data = config['artifacts']['featurize_test_data']

    art_dir_name = config['artifacts']['ARTIFACTS_DIR']
    # print(art_dir_name)
    prepare_dir_name = config['artifacts']['PREPARE_DIR']
    # print(prepare_dir_name)
    train_file_name = config['artifacts']['TRAIN_DATA']
    test_file_name =  config['artifacts']['TEST_DATA']

    prepare_dir_path = os.path.join(art_dir_name,prepare_dir_name)
    
    train_file_path = os.path.join(prepare_dir_path,train_file_name)
    test_file_path = os.path.join(prepare_dir_path,test_file_name)


    # create feature vector dir
    featurize_dir_path = os.path.join(art_dir,featurize_dir)
    create_directories([featurize_dir_path])

    featurize_train_data_path = os.path.join(featurize_dir_path,featurize_train_data)
    featurize_test_data_path = os.path.join(featurize_dir_path,featurize_test_data)

    max_features = params['featurize']['max_feature']
    ngrams = params['featurize']['ngrams']

    df_train = get_df(train_file_path)
    train_words = df_train.text.str.lower().values.astype("U")
    # print(type(train_words))
    # print(train_words[:20])
    bag_of_words = CountVectorizer(stop_words="english",max_features=max_features,ngram_range=(1,ngrams))
    bag_of_words.fit(train_words)
   # print(bag_of_words.vocabulary_)
    train_word_binary_metrix = bag_of_words.transform(train_words)
    featurize.save_metrix(df_train,train_word_binary_metrix,featurize_train_data_path)


    

if __name__=="__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--config", "-c", default="configs/config.yaml")
    args.add_argument("--params", "-p", default="params.yaml")
    parsed_args = args.parse_args()

    

    try:
        logging.info("\n********************")
        logging.info(f">>>>> stage {STAGE} started <<<<<")
        main(config_path=parsed_args.config, params_path=parsed_args.params)
        logging.info(f">>>>> stage {STAGE} completed!<<<<<\n")
    except Exception as e:
        logging.exception(e)
        raise e