from __future__ import annotations

from pathlib import Path

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension


ROOT = Path(__file__).resolve().parent


setup(
    name="gomoku-zero-ai",
    ext_modules=[
        CppExtension(
            name="gomoku_zero._mcts_cpp",
            sources=[str(ROOT / "gomoku_zero" / "cpp" / "mcts_extension.cpp")],
            extra_compile_args={
                "cxx": ["/O2", "/std:c++17"] if __import__("platform").system() == "Windows" else ["-O3", "-std=c++17"],
            },
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
