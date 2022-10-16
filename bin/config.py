import argparse

def parse_args():
    parser = argparse.ArgumentParser(description = 'Meta-Spec')
    
    ## Data settings
    parser.add_argument('--microbe', type=str, default='../data/train_microbe_data.csv')
    parser.add_argument('--host', type=str, default='')
    parser.add_argument('--label', type=str, default='../data/train_labels.csv')
    parser.add_argument('--o', type=str, default='out/')   
    
    ## Model settings
    parser.add_argument('--task_num', type=int, default=5)
    parser.add_argument('--embedding_size', type=int, default=128)
    parser.add_argument('--n_expert', type=int, default=9)
    parser.add_argument('--hidden_units', type=tuple, default=(256, 128))
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--n_epoch', type=int, default=8)
    
    ## SHAP settings
    parser.add_argument('--is_plot', type=bool, default=True)
    parser.add_argument('--max_plot', type=int, default=60)
    
    return parser.parse_args()