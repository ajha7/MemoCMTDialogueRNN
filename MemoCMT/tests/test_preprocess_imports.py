import ast
import os

PREPROCESS = os.path.join(os.path.dirname(__file__), "..", "scripts", "preprocess.py")


def test_moviepy_is_not_imported_at_module_level():
    # moviepy>=2 removed moviepy.editor, and only the MELD/mp4 paths need it. A top-level
    # import would stop IEMOCAP preprocessing on a fresh Colab install.
    tree = ast.parse(open(PREPROCESS).read())
    top_level = [n for n in tree.body if isinstance(n, (ast.Import, ast.ImportFrom))]
    modules = [n.module for n in top_level if isinstance(n, ast.ImportFrom)]
    modules += [a.name for n in top_level if isinstance(n, ast.Import) for a in n.names]
    assert not any(m and m.startswith("moviepy") for m in modules)
