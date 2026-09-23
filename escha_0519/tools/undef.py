"""Report names used but never bound in a module (module scope + function bodies).
Deliberately conservative: only flags names that appear nowhere as a binding."""

import ast
import builtins
import sys

BUILTINS = set(dir(builtins))


def bound_names(tree):
    out = set()
    for n in ast.walk(tree):
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            out.add(n.name)
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)):
                a = n.args
                for arg in list(a.args) + list(a.posonlyargs) + list(a.kwonlyargs):
                    out.add(arg.arg)
                if a.vararg:
                    out.add(a.vararg.arg)
                if a.kwarg:
                    out.add(a.kwarg.arg)
        elif isinstance(n, (ast.Import, ast.ImportFrom)):
            for al in n.names:
                out.add(al.asname or al.name.split(".")[0])
        elif isinstance(n, ast.Name) and isinstance(n.ctx, (ast.Store, ast.Del)):
            out.add(n.id)
        elif isinstance(n, ast.ExceptHandler) and n.name:
            out.add(n.name)
        elif isinstance(n, (ast.Global, ast.Nonlocal)):
            out.update(n.names)
        elif isinstance(n, ast.arg):
            out.add(n.arg)
        elif isinstance(n, (ast.comprehension,)):
            pass
        elif isinstance(n, ast.withitem) and n.optional_vars is not None:
            for sub in ast.walk(n.optional_vars):
                if isinstance(sub, ast.Name):
                    out.add(sub.id)
    return out


bad = 0
for f in sys.argv[1:]:
    tree = ast.parse(open(f).read())
    bound = (
        bound_names(tree)
        | BUILTINS
        | {"__name__", "__file__", "__doc__", "self", "cls"}
    )
    used = {
        n.id
        for n in ast.walk(tree)
        if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)
    }
    missing = sorted(used - bound)
    if missing:
        print(f"{f}: {missing}")
        bad += len(missing)
print("undefined names:", bad)
