import ast
import os
import random
import string
from glob import iglob
from typing import Any

from tqdm import tqdm


def random_string_generator(size=6, chars=string.ascii_letters + string.digits):
    return random.choice(string.ascii_letters) + "".join(random.choice(chars) for _ in range(size - 1))


class BugInserter(ast.NodeTransformer):

    def visit_Assign(self, node: ast.Assign) -> Any:
        if random.random() < 0.5:
            for target in node.targets:
                target.id = random_string_generator(random.choice(range(3, 10)))
        return node


if __name__ == "__main__":
    source_folder, source_extension = "./raw_repos", ".py"
    save_folder, save_extension = "./bug", ".bug.py"


    def dest_to_source_file(path):
        return f"{source_folder}{path[len(save_folder):-len(save_extension)]}{source_extension}"


    file_paths = set(iglob(os.path.join(source_folder, "**", f"*{source_extension}"), recursive=True))
    processed_files = set(iglob(os.path.join(save_folder, "**", f"*{save_extension}"), recursive=True))
    # the following is ugly but seem to be the fastest way how to achieve the goal
    files_to_process = file_paths.difference({dest_to_source_file(file) for file in processed_files} )

    for file_path in tqdm(files_to_process):
        try:
            with open(file_path) as file:
                code = file.read()
            tree = ast.parse(code)
        except (SyntaxError, IsADirectoryError):  # no matter what happens continue
            continue
        optimizer = BugInserter()
        tree = optimizer.visit(tree)

        # save modified file
        dirname = os.path.dirname(file_path.replace(source_folder, save_folder))
        os.makedirs(dirname, exist_ok=True)
        filename = os.path.basename(file_path)[:len(source_extension)] + save_extension
        with open(os.path.join(dirname, filename), "w") as file:
            file.write(ast.unparse(tree))
