#!/bin/bash

python ../bin/meta_spec_train.py --microbe train_microbe_data.csv --host train_hosts_data.csv --label train_labels.csv --o out

python ../bin/meta_spec_test.py --microbe test_microbe_data.csv --host test_hosts_data.csv --o out

python ../bin/meta_spec_imp.py --microbe train_microbe_data.csv --host train_hosts_data.csv --label train_labels.csv --o out

python ../bin/meta_spec_get_msi.py --microbe test_microbe_data.csv --host test_hosts_data.csv --o out --is_plot True --max_plot 30