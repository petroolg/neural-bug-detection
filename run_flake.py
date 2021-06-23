import json
import os
import subprocess
from glob import iglob


def run_flake(file_path):
    result = subprocess.run(["flake8", "--format=json", "--select=F821,F841", file_path], stdout=subprocess.PIPE)
    return result.stdout


if __name__ == '__main__':
    source_folder = "./bug"
    save_folder = "./flake8"
    for file_path in iglob(os.path.join(source_folder, '**', '*.py'), recursive=True):
        flake_report = run_flake(file_path)
        print(file_path)
        flake_report_json = json.loads(flake_report)[file_path]
        dirname = os.path.dirname(file_path.replace(source_folder, save_folder))
        os.makedirs(dirname, exist_ok=True)
        filename = os.path.basename(file_path)[:-3] + ".json"
        with open(os.path.join(dirname, filename), "w") as file:
            json.dump(flake_report_json, file)
