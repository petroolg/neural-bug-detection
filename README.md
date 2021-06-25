# neural-bug-detection
A graph neural network designed to detect and miss bugs in python code.

## Requirements installation

The project is compatible with Python>=3.9. Make sure you have the correct version. Then from the terminal run:
```shell script
pip install -r requrements.txt
```

## Data preprocessing

1) Get raw data (python scripts), e.g. from here [Py150k](https://www.sri.inf.ethz.ch/py150)
> :warning: **From this point on all commands should necessary be executed from the data root directory. Data root directory stores `./raw_repos` folder with your raw data (.py files).**

2) Execute `bug_insertion.py` from the data root directory. Make sure you have python>=3.9.

   > The script expects to find repos in `./raw_repos` directory.
   ```shell script
   python <path-to-neural-bug-insertion>/data_preprocessing/bug_insertion.py
   ```
   saves the output to `./bug` directory keeping 
   the original directory structure.
   
3) Execute `run_flake.py` from the data root directory.
   ```shell script
   python <path-to-neural-bug-insertion>/data_preprocessing/run_flake.py
   ```
   The script runs flake8 for every file in `./bug` directory and saves the output to `./flake8` directory keeping 
   the original directory structure.
   
   A shell command equivalent to the one used in `run_flake.py`:
   ```shell script
     flake8 --format=json --select=F821,F841 <file_path>
   ```
      
4) Clone forked [typilus repo](https://github.com/petroolg/typilus) and build Docker from 
[`src/data_preparation/Dockerfile`](https://github.com/petroolg/typilus/blob/master/src/data_preparation/Dockerfile) 
following the steps from
[`README.md`](https://github.com/petroolg/typilus/blob/master/src/data_preparation/README.md) 
manual located in the same folder. 

    Once you have the `typilus-env` container up and running, execute the following command to finish data preparation.
    ```bash
    bash scripts/prepare_data_custom.sh
    ```
   The script processes files from `./bug` directory. It creates a dataset with labelled AST's (abstract syntax tree) 
   extracted from python scripts with bugs introduced earlier in 2. step. Simultaneously it links flake8 error 
   codes with corresponding AST nodes for each file.  
   
## Graph neural network training

5) Construct pytorch-geometric dataset by running the following command from the data root directory.

    ```shell script
    python <path-to-neural-bug-insertion>/graph_embedding.py 
    ```
   The script processes files from "./graph-dataset" directory. The output is saved to `./processed` directory.
   
6) Running training script: from the data root directory execute the following command:
    ```shell script
    python <path-to-neural-bug-insertion>/nn_training.py 
    ```
   The training script 1) loads the dataset created in the previous step, 2)constructs graph neural network and 
   3) runs training loop for 200 epochs. After each epoch the script plots a confusion matrix computed from test set data.
