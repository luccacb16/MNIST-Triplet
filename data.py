import subprocess
import os

def download_dataset(data_path: str = './data/'):
    train_path = os.path.join(data_path, 'mnist_train.csv')
    test_path = os.path.join(data_path, 'mnist_test.csv')
    
    # Verifica se o data_path existe
    if not os.path.exists(data_path):
        os.makedirs(data_path)

    # Verifica se os arquivos já existem
    if os.path.exists(train_path) and os.path.exists(test_path):
        return

    # Download do dataset MNIST do Kaggle
    print("Baixando o dataset MNIST...")
    subprocess.run(f"kaggle datasets download -d oddrationale/mnist-in-csv -p {data_path}", shell=True, check=True)

    # Descompactação do arquivo baixado
    print("Descompactando o arquivo...")
    subprocess.run(f"unzip {data_path}/mnist-in-csv.zip -d {data_path}", shell=True, check=True)

    # Remoção do arquivo zip
    print("Removendo o arquivo zip...")
    subprocess.run(f"rm {data_path}/mnist-in-csv.zip", shell=True, check=True)