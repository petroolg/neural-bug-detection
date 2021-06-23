import ast
import os
import random
import string
from glob import iglob
from typing import Any


def id_generator(size=6, chars=string.ascii_letters + string.digits):
    return random.choice(string.ascii_letters) + ''.join(random.choice(chars) for _ in range(size - 1))


class BugInserter(ast.NodeTransformer):

    def visit_Assign(self, node: ast.Assign) -> Any:
        if random.random() < 0.3:
            for target in node.targets:
                target.id = id_generator(5)
        return node


if __name__ == '__main__':
    source_folder = "./raw_repos"
    save_folder = "./bug"
    for file_path in iglob(os.path.join(source_folder, '**', '*.py'), recursive=True):
        print(file_path)
        with open(file_path) as file:
            code = file.read()
        try:
            tree = ast.parse(code)
        except SyntaxError:
            continue
        optimizer = BugInserter()
        tree = optimizer.visit(tree)

        dirname = os.path.dirname(file_path.replace(source_folder, save_folder))
        os.makedirs(dirname, exist_ok=True)
        filename = os.path.basename(file_path)[:-3] + ".bug.py"
        with open(os.path.join(dirname, filename), "w") as file:
            file.write(ast.unparse(tree))
