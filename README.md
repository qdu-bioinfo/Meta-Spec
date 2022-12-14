# **Meta-Spec**

## Contents

- [Introduction](#introduction)
- [Package requirement](#package-requirement)
- [Installation](#installation)
- [Model training](#model-training)
- [Classification ](#classification )
- [Meta-Spec importance](#meta-spec-importance)
- [Citation](#citation)
- [Contact](#contact)

## Introduction

Meta-Spec is a microbiome multi-label disease classification tool based on explainable deep learning. Meta-Spec is capable to detect multiple diseases simultaneously by integrating genotype data (microbiome features) and phenotype data (host variables). Meta-Spec can also achieve high performance on regular single-label classification tasks.

## Package requirement

- torch==1.9.0+cu111
- deepctr-torch==0.2.7
- shap==0.35.0

## Installation
```
sh init.sh
```
Then all tools are located at ‘bin’ folder:
```
meta_spec_train.py // For model training
meta_spec_test.py // For disease prediction
meta_spec_imp.py // To add Meta-Spec Importance (MSI) in a model
meta_spec_get_msi.py // To calculate Meta-Spec Importance (MSI)
```
## Model training
To train Meta-Spec model, two files are required as input including microbial features (e.g. ASV, OTU, species, gene, etc.) and diseases labels. Host variables (e.g. age, BMI, etc.) are optional, which can largely improve the overall performance. 
a. Microbial features (required)
| **SampleID** | **asv_1**  | **asv_2**| **...**| **asv_m**|
| ---------- | --------- | --------- | --------- | --------- |
| Sample_1        | 0.1      | 0.002    | ...      | 0      |
| Sample_2        | 0.05     | 0.01     | ...      | 0.01   |
| ...             | ...      | ...      | ...      | ...    |
| Sample_N        | 0.07     | 0.03     | ...      | 0.01   |

b. Disease labels (required)
| **SampleID** | **ibs**  | **thyroid**| **...**| **disease_k**|
| ---------- | --------- | --------- | --------- | --------- |
| Sample_1        | 0      | 1      | ...      | 0      |
| Sample_2        | 0      | 0      | ...      | 0      |
| ...             | ...    | ...    | ...      | ...    |
| Sample_N        | 1      | 1      | ...      | 0      |

c. Host variables (optional, but recommended)
| **SampleID** | **age**  | **bmi**| **...**| **variable_x**|
| ---------- | --------- | --------- | --------- | --------- |
| Sample_1      | 5      | 2      | ...      | 1      |
| Sample_2      | 2      | 1      | ...      | 0      |
| ...           | ...    | ...    | ...      | ...    |
| Sample_N      | 7      | 3      | ...      | 1      |

You can assign microbial features, diseases label and host variables by '--microbe' ,'--label' and '--host', respectively. In addition, you can specify the output path of model '--o'. We set an example dataset in ‘data’ folder for quick start. To train a Meta-Spec model: 
```
cd data

python ../bin/meta_spec_train.py --microbe train_microbe_data.csv --host train_hosts_data.csv --label train_labels.csv --o out
```

## Classification 
Then, you can predict the status of microbiomes using the model generated by the training procedure. We set an example dataset in ‘data/’ folder for quick start.
```
python ../bin/meta_spec_test.py --microbe test_microbe_data.csv --host test_hosts_data.csv --o out
```

## Meta-Spec importance
To calculate the Meta-Spec Importance (MSI), you need to make an additional training procedure to prepare for MSI. 
```
python ../bin/meta_spec_imp.py --microbe train_microbe_data.csv --host train_hosts_data.csv --label train_labels.csv --o out
```
After that, you can obtain MSI values. If the '--is_plot' is enabled, bar charts of MSI will be plotted.
```
python ../bin/meta_spec_get_msi.py --microbe test_microbe_data.csv --host test_hosts_data.csv --o out --is_plot True --max_plot 30
```
For convenience, you can run the processes above by running the example.sh in folder 'data/'. 
```
cd data
chmod a+x example.sh
./example.sh
```
Other settings can be seen in the config.py.

## Citation


## Contact
All problems please contact Meta-Spec development team: 
**Xiaoquan Su**&nbsp;&nbsp;&nbsp;&nbsp;Email: suxq@qdu.edu.cn
