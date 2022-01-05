import xml.etree.cElementTree as ET
import re
import random
import logging

split = 0.20

def write_train_test_data(in_file,train_file,test_file,target):
    count = 0
    for line in in_file:
        try:
            fd_out = train_file if random.random() > split else test_file

            attr = ET.fromstring(line).attrib
            id = attr.get("Id")
            label = 1 if target in attr.get("Tags","") else 0
            title = re.sub(r"\s+"," ",attr.get("Title","")).strip()
            body =  re.sub(r"\s+"," ",attr.get("Body","")).strip()
            text = title + " " + body

            fd_out.write(f"{id}\t{label}\t,{text}\n")
            count+=1
        except Exception as e:
            msg = f"skipping the broken line {count}:{e}\n"
            logging.exception(msg)


