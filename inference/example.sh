#!/bin/bash

python meta_spec_train.py --microbe '../data/train_microbe_data.csv' --host '../data/train_hosts_data.csv' --label '../data/train_labels.csv' --o 'res/' --m 'meta_spec.model'

python meta_spec_test.py --microbe '../data/test_microbe_data.csv' --host '../data/test_hosts_data.csv' --o 'res/' --m 'meta_spec.model' --disease_name 'ibs thyroid migraine autoimmune lung_disease'

python meta_spec_imp.py --microbe '../data/train_microbe_data.csv' --host '../data/train_hosts_data.csv' --label '../data/train_labels.csv' --o 'res/' --m 'msi.model'

python get_msi.py --microbe '../data/test_microbe_data.csv' --host '../data/test_hosts_data.csv' --o 'res/' --m 'msi.model' --disease_name 'ibs thyroid migraine autoimmune lung_disease' --is_plot True --max_plot 30