import argparse

def get_arguments():

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', dest='config', help='Settings of each dataset in yaml format')
    parser.add_argument('--shots', dest='shots', type=int, default=1, help='Number of shots')
    parser.add_argument('--root_path', dest='root_path', default='./data', help='Path to the root folder of all datasets')
    
    parser.add_argument('--model_name', dest='model_name', default='datvilc', help='The name of the model to train. It can be either datvilc or datvil' )
    parser.add_argument('--per_sample_train', dest='per_sample_train', type=int , default=10, help='The number of per-sample training epochs in DATViL')
    parser.add_argument('--plus_residual', dest='plus_residual', type=int , default=1, help='Add residual-based adatpers')
    parser.add_argument('--plus_transform', dest='plus_transform', type=int , default=1, help='Add transormer-based adapters')

    args = parser.parse_args()

    return args