# neural-bug-detection
A graph neural network designed to detect and miss bugs in python code.

### Requirements installation

The project is compatible with Python>=3.9. Make sure you have the correct version. Then from the terminal run:
```shell script
pip install -r requrements.txt
```

### Dataset construction
In order to construct the dataset, we need:

1) Get raw data (python scripts), e.g. from here [Py150k](https://www.sri.inf.ethz.ch/py150)
2) Execute `bug_insertion.py` from the root directory.
   ```shell script
   python bug_insertion.py
   ```
3) Execute `run_flake.py` from the root directory.
   ```shell script
   python run_flake.py
   ```
4) Clone https://github.com/petroolg/typilus and build Docker from `src/data_preparation/Dockerfile` following the 
`README.md` manual located at the same folder. 

    Once you have the `typilus-env` container up and running, execute the following line to finish data preparation.
    ```bash
    bash scripts/prepare_data_custom.sh
    ```
