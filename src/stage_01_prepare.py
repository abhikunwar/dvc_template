import argparse
import os
import shutil
from tqdm import tqdm
import logging
from src.utils.common import read_yaml, create_directories
from src.utils.data_mgmt import write_train_test_data
import random


STAGE = "STAGE_NAME" ## <<< change stage name 

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

    art_dir_name = config['artifacts']['ARTIFACTS_DIR']
    # print(art_dir_name)
    prepare_dir_name = config['artifacts']['PREPARE_DIR']
    # print(prepare_dir_name)
    train_file_name = config['artifacts']['TRAIN_DATA']
    test_file_name =  config['artifacts']['TEST_DATA']

    prepare_dir_path = os.path.join(art_dir_name,prepare_dir_name)
    create_directories([prepare_dir_path])

    train_file_path = os.path.join(prepare_dir_path,train_file_name)
    test_file_path = os.path.join(prepare_dir_path,test_file_name)

    data_dir_name = config['source_data']['data_dir']
    data_file_name = config['source_data']['data_file']
    data_file_path = os.path.join(data_dir_name,data_file_name)

    with open(data_file_path,encoding="utf8") as in_file:
        with open(train_file_path,"w",encoding="utf8") as train_file:
            with open(test_file_path,"w",encoding = "utf8") as test_file:
                write_train_test_data(in_file,train_file,test_file,"python")

    
    


if __name__ == '__main__':
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