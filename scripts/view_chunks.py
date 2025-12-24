from pathlib import Path
import sys

from coderetrx import Codebase
from coderetrx.static.codebase import CodeChunk
from coderetrx.retrieval.factory import CodebaseFactory
from coderetrx.retrieval.smart_codebase import SmartCodebaseSettings


def load_coderetrx_codebase(codebase_path: str) -> Codebase:
    abs_repo_path = Path(codebase_path).resolve()

    smart_settings = SmartCodebaseSettings()
    smart_settings.symbol_codeline_embedding = False
    smart_settings.keyword_embedding = False
    smart_settings.symbol_content_embedding = False
    smart_settings.symbol_name_embedding = False

    codebase = CodebaseFactory.new(
        str(abs_repo_path),
        abs_repo_path,
        smart_settings,
    )
    return codebase


def show_code_chunks(codebase: Codebase, file_name: str):
    for chunk in codebase.get_splited_distinct_chunks(100):
    # for chunk in codebase.all_chunks:
        if str(chunk.src.path) != file_name:
            continue
        print("Type:", chunk.type)
        print()
        print(
            chunk.ast_codeblock(
                show_line_numbers=True,
                zero_based_line_numbers=False,
                show_imports=True,
            )
        )
        print("=" * 40)


def main():
    codebase_path = sys.argv[1]
    codebase = load_coderetrx_codebase(codebase_path)

    file_name = sys.argv[2]
    show_code_chunks(codebase, file_name)


if __name__ == "__main__":
    main()
