#!/bin/bash -x  

cd src

python gen_train_data.py

python gen_features.py partdata
python lgb_model.py partdata

python gen_features.py fulldata
python lgb_model.py fulldata

python rule_and_merge.py

