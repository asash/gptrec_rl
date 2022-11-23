ENV_NAME='aprec'
CUR_DIR=`pwd`
DIR="$(dirname "$CUR_DIR")"
conda create -y -c pytorch -c conda-forge --name $ENV_NAME python=3.9.12 cudnn=8.4.1.50 cudatoolkit=11.7.0
conda install -n $ENV_NAME -y pytorch-gpu=1.12.1
conda install -n $ENV_NAME -y tensorflow-gpu=2.10.0
conda install -n $ENV_NAME -y gh=2.1.0 
conda install -n $ENV_NAME -y expect=5.45.4
conda env config vars set -n $ENV_NAME PYTHONPATH=$DIR
conda env config vars set -n $ENV_NAME LD_LIBRARY_PATH="$LD_LIBRARY_PATH:~/anaconda3/envs/$ENV_NAME/lib"

~/anaconda3/envs/$ENV_NAME/bin/pip3 install "jupyter>=1.0.0" "tqdm>=4.62.3" "requests>=2.26.0"\
               "pandas>=1.3.3" "lightfm>=1.16" "scipy>=1.6.0" "tornado>=6.1"\
               "numpy>=1.19.5" "scikit-learn>=1.0" "lightgbm>=3.3.0"\
               "mmh3>=3.0.0" "matplotlib>=3.4.3" "seaborn>=0.11.2"\
               "jupyterlab>=3.2.2" "telegram_send>=0.25" "recbole==1.0.1"\
               "wget>=3.2" "pytest>=7.1.2" "pytest-forked>=1.4.0" "setuptools==59.5.0"\
               "transformers>=4.24.0"\
               "faiss_gpu>=1.7.2" "dill>=0.3.6" "multiprocessing_on_dill>=3.5.0a4"
