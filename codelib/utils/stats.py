from ._extras import require_extra

require_extra("tiktoken", "stats")

from pydantic import BaseModel
from collections import Counter
from typing import Dict, List, ClassVar
from pathlib import Path
import tiktoken
from concurrent.futures import ThreadPoolExecutor, as_completed

from codelib.static.codebase import Codebase, File, CodeChunk, ChunkType
from codelib.static.codebase.languages import get_language


class ChunkStats(BaseModel):
    chunk_id: str
    file_path: str
    num_lines: int
    num_tokens: int

    # Class variable for the tokenizer
    _tokenizer: ClassVar = tiktoken.encoding_for_model("gpt-4o")

    @classmethod
    def from_chunk(cls, chunk: CodeChunk, strict: bool = False) -> "ChunkStats":
        lines = chunk.lines()
        text = "\n".join(lines)

        SAMPLE_SIZE = 100_000
        if not strict and len(text) > SAMPLE_SIZE:
            tokens = len(cls._tokenizer.encode(text[:SAMPLE_SIZE])) * (
                len(text) / SAMPLE_SIZE
            )
        else:
            tokens = len(cls._tokenizer.encode(text))

        return cls(
            chunk_id=chunk.id,
            file_path=str(chunk.src.path),
            num_lines=len(lines),
            num_tokens=int(tokens),
        )


class FileStats(BaseModel):
    file_path: str
    num_chunks: int
    num_lines: int
    num_tokens: int

    @classmethod
    def from_file(cls, file: File) -> "FileStats":
        with ThreadPoolExecutor() as executor:
            # Process chunks in parallel
            future_to_chunk = {
                executor.submit(ChunkStats.from_chunk, chunk): chunk
                for chunk in file.chunks
                if chunk.type in [ChunkType.PRIMARY, ChunkType.IMPORT]
            }
            chunk_stats = [future.result() for future in as_completed(future_to_chunk)]

        return cls(
            file_path=str(file.path),
            num_chunks=len(file.chunks),
            num_lines=len(file.lines),
            num_tokens=sum(stat.num_tokens for stat in chunk_stats),
        )


class CodebaseStats(BaseModel):
    codebase_id: str
    num_chunks: int
    num_files: int
    num_lines: int
    num_tokens: int
    language_distribution: Dict[str, int]  # Maps language name to line count
    primary_language: str  # The language with the most lines of code
    file_stats: List[FileStats]
    chunk_stats: List[ChunkStats]

    @classmethod
    def from_codebase(cls, codebase: Codebase) -> "CodebaseStats":
        with ThreadPoolExecutor() as executor:
            # Process files in parallel
            future_to_file = {
                executor.submit(FileStats.from_file, file): file_path
                for file_path, file in codebase.source_files.items()
            }
            file_stats = [future.result() for future in as_completed(future_to_file)]

        # Calculate language distribution by line count
        language_counts = Counter()
        for file_path, file in codebase.source_files.items():
            lang = get_language(Path(file_path))
            if lang:
                language_counts[lang] += len(file.lines)

        # Determine primary language (language with most lines)
        primary_language = max(language_counts, key=language_counts.get) if language_counts else "Unknown"

        # Collect all chunk stats
        chunk_stats = []
        for file in codebase.source_files.values():
            for chunk in file.chunks:
                if chunk.type in [ChunkType.PRIMARY, ChunkType.IMPORT]:
                    chunk_stats.append(ChunkStats.from_chunk(chunk))

        return cls(
            codebase_id=codebase.id,
            num_chunks=len(codebase.all_chunks),
            num_files=len(codebase.source_files),
            num_lines=sum(stat.num_lines for stat in file_stats),
            num_tokens=sum(stat.num_tokens for stat in file_stats),
            language_distribution=dict(language_counts),
            primary_language=primary_language,
            file_stats=file_stats,
            chunk_stats=chunk_stats,
        )


if __name__ == "__main__":
    import argparse
    from textwrap import dedent

    parser = argparse.ArgumentParser()
    parser.add_argument("target", type=str, help="Path to codebase")
    args = parser.parse_args()

    codebase = Codebase.new("tmp", Path(args.target))
    codebase.init_chunks()
    stats = CodebaseStats.from_codebase(codebase)
    print(
        dedent(f"""
        Codebase Stats for {args.target}:
        - Number of chunks: {stats.num_chunks}
        - Number of files: {stats.num_files}
        - Number of lines: {stats.num_lines}
        - Number of tokens: {stats.num_tokens}
        - Primary language: {stats.primary_language}
        - Language distribution: {stats.language_distribution}
        """)
    )
