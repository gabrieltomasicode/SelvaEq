"""
Utilitário de limpeza de cache (__pycache__ e .pyc), reaproveitado integralmente
de `SelvaSonicScripts/cleancache.py`.
"""

import os
import shutil
import stat


def handle_remove_readonly(func, path, exc_info):
    """Remove o atributo 'somente leitura' de arquivos bloqueados e tenta excluir novamente."""
    try:
        os.chmod(path, stat.S_IWRITE)
        func(path)
    except Exception:
        pass


def remove_pycache_and_pyc(root_dir=None):
    """Remove todas as pastas __pycache__ e arquivos .pyc recursivamente."""
    if root_dir is None:
        root_dir = os.path.dirname(os.path.abspath(__file__))

    for dirpath, dirnames, filenames in os.walk(root_dir):
        if "__pycache__" in dirnames:
            pycache_path = os.path.join(dirpath, "__pycache__")
            try:
                shutil.rmtree(pycache_path, onerror=handle_remove_readonly)
            except Exception:
                pass
            dirnames.remove("__pycache__")

        for filename in filenames:
            if filename.endswith(".pyc"):
                pyc_path = os.path.join(dirpath, filename)
                try:
                    os.chmod(pyc_path, stat.S_IWRITE)
                    os.remove(pyc_path)
                except Exception:
                    pass


if __name__ == "__main__":
    remove_pycache_and_pyc()
