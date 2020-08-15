import argparse
import os
import pandas as pd
import boto3
import joblib
import json
import pickle as pkl
#specific imports
from sagemaker_containers import entry_point
from sagemaker_xgboost_container.data_utils import get_dmatrix
from sagemaker_xgboost_container import distributed
import sagemaker_xgboost_container.encoder as xgb_encoders

from xgboost import DMatrix
import xgboost as xgb
        
def model_fn(model_dir):
    print("Loading model from file.")
    with open(os.path.join(model_dir, 'xgboost-model'), "rb") as f:  
        booster = pkl.load(f)
    return booster

def _xgb_train(params, dtrain, evals, num_boost_round, model_dir, is_master):
    booster = xgb.train(params=params,
                        dtrain=dtrain,
                        evals=evals,
                        num_boost_round=num_boost_round)
    if is_master:
        model_location = model_dir + '/xgboost-model'
        pkl.dump(booster, open(model_location, 'wb'))
        print("Stored trained model at {}".format(model_location))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--num-round', type=int, default=100)
    parser.add_argument('--max-depth', type=int, default=6)
    parser.add_argument('--eta', type=float, default=0.3)
    parser.add_argument('--gamma', type=float, default=0.0)
    parser.add_argument('--objective', type=str, default='binary:hinge')
    parser.add_argument('--booster', type=str, default='gblinear')
    parser.add_argument('--tree-method', type=str, default='auto')
    parser.add_argument('--eval-metric', type=str, default='rmse')
    print("Done parsing hyperparameters.")
    
    # Sagemaker specific arguments. Defaults are set in the environment variables.
    
    parser.add_argument('--output_data_dir', type=str, default=os.environ.get('SM_OUTPUT_DATA_DIR'))
    parser.add_argument('--model_dir', type=str, default=os.environ.get('SM_MODEL_DIR'))
    parser.add_argument('--train', type=str, default=os.environ.get('SM_CHANNEL_TRAIN'))
    parser.add_argument('--validation', type=str, default=os.environ.get('SM_CHANNEL_VALIDATION'))
    parser.add_argument('--sm_hosts', type=str, default=os.environ.get('SM_HOSTS'))
    parser.add_argument('--sm_current_host', type=str, default=os.environ.get('SM_CURRENT_HOST'))
    parser.add_argument('--test', type=str, default=os.environ.get('SM_CHANNEL_TEST'))
    #args = parser.parse_args()
    args, _ = parser.parse_known_args()
    print("Done parsing arguments.")
   
    dtrain = get_dmatrix(args.train, 'csv') 
    dval = get_dmatrix(args.validation, 'csv')
    if dval is not None:
        watchlist = [(dtrain, 'train'), (dval, 'validation')]  
    else: 
        watchlist= [(dtrain, 'train')]
    #watchlist = [(dtrain, 'train'),(dval, 'validation')]
    print("Done defining data and host information.")
    
    sm_hosts = json.loads(args.sm_hosts)
    sm_current_host = args.sm_current_host
    print("Done loading hosts.")
   
    train_hp = {
        'n_estimators':args.num_round, 
        'max_depth': args.max_depth,
        'eta': args.eta,
        'gamma': args.gamma,
        'objective': args.objective,
        'eval_metric':args.eval_metric,
        'booster':args.booster,
        'tree_method':args.tree_method
    }
    print("Done setting up hyperparams")
    xgb_train_args = dict(
        params=train_hp,
        dtrain=dtrain,
        evals=watchlist,
        num_boost_round=args.num_round,
        model_dir=args.model_dir
    )
    print("Done training")
    
    if len(sm_hosts) > 1:
        # Wait until all hosts are able to find each other
        entry_point._wait_hostname_resolution()

        # Execute training function after initializing rabit.
        distributed.rabit_run(
            exec_fun=_xgb_train,
            args=xgb_train_args,
            include_in_training=(dtrain is not None),
            hosts=sm_hosts,
            current_host=sm_current_host,
            update_rabit_args=True
        )
    else:
        # If single node training, call training method directly.
        if dtrain:
            xgb_train_args['is_master'] = True
            _xgb_train(**xgb_train_args)
        else:
            raise ValueError("Training channel must have data to train model.")
