import scipy
from scipy.sparse import csr_matrix
import scipy.sparse as sparse
import numpy as np
import logging
import joblib

def save_metrix(df,metrix,out_path):
    id_metrix = sparse.csr_matrix(df.id.astype(np.int64)).T 
    label_metrix = sparse.csr_matrix(df.label.astype(np.int64)).T 

    result = sparse.hstack([id_metrix,label_metrix,metrix],format = "csr")
    # print(result.toarray())

    msg = f"the output metrix {out_path} of size {result.shape} and data type {result.dtype}"
    logging.info(msg)

    joblib.dump(result,out_path)

