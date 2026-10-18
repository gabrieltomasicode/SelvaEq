"""
Executor unificado da suíte de testes unitários e de integração do SelvaEQ.
"""

import os
import sys
import unittest


def run_all_tests() -> bool:
    root_dir = os.path.dirname(os.path.abspath(__file__))
    if root_dir not in sys.path:
        sys.path.insert(0, root_dir)

    loader = unittest.TestLoader()
    suite = loader.discover(start_dir=os.path.join(root_dir, "tests"), top_level_dir=root_dir)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
