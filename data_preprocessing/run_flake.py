import json
import os
import subprocess
from glob import iglob

from tqdm import tqdm


def run_flake(file_path: str) -> bytes:
    result = subprocess.run(["flake8", "--format=json", "--select=F821,F841", file_path], stdout=subprocess.PIPE)
    return result.stdout


if __name__ == "__main__":
    source_folder, source_extension = "./bug", ".bug.py"
    save_folder, save_extension = "./flake8", ".bug.json"


    def dest_to_source_file(path):
        return f"{source_folder}{path[len(save_folder):-len(save_extension)]}{source_extension}"


    file_paths = set(iglob(os.path.join(source_folder, "**", f"*{source_extension}"), recursive=True))
    processed_files = set(iglob(os.path.join(save_folder, "**", f"*{save_extension}"), recursive=True))
    files_to_process = file_paths - {dest_to_source_file(file) for file in processed_files}

    for file_path in tqdm(files_to_process):
        flake_report = run_flake(file_path)
        try:
            flake_report_json = json.loads(flake_report)[file_path]
        except KeyError:
            continue

        # save modified file
        dirname = os.path.dirname(save_folder + file_path[len(source_folder):])
        os.makedirs(dirname, exist_ok=True)
        filename = os.path.basename(file_path)[:-len(source_extension)] + save_extension
        with open(os.path.join(dirname, filename), "w") as file:
            json.dump(flake_report_json, file)
