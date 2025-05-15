from pathlib import Path
from typing import Dict, List, Literal, Optional, Self, Union

from pydantic import BaseModel, Field

from .codebase import (
    ChunkType,
    Codebase,
    CodeChunk,
    CodeHunk,
    File,
    hunk_uuid,
    IDXSupportedTag,
)

FileType = Literal["source", "dependency"]

# TODO: include tree-sitter node id in code chunk (each chunk must correspond to a node)


class CodeChunkModel(BaseModel):
    type: ChunkType
    start_line: int
    end_line: int
    start_column: int
    end_column: int
    src: str  # Path to source file
    uuid: str
    content: Optional[str] = None
    ts_node_id: int  # Tree-sitter node ID
    tag: Optional[IDXSupportedTag] = None

    @classmethod
    def from_chunk(
        cls, chunk: CodeChunk, include_content: bool = False
    ) -> "CodeChunkModel":
        return cls(
            type=chunk.type,
            start_line=chunk.start_line,
            end_line=chunk.end_line,
            start_column=chunk.start_column,
            end_column=chunk.end_column,
            src=str(chunk.src.path),
            content=chunk.code() if include_content else None,
            ts_node_id=chunk.ts_root.id,
            tag=chunk.tag,
            uuid=str(chunk.uuid),
        )

    def to_chunk(self, src_file: File) -> CodeChunk:
        # Initialize tree-sitter and get the root node
        ts_state = src_file.ts()
        tree = ts_state.parser.parse(src_file.content.encode())
        # Try to find the node with matching ID, fallback to root node if not found
        ts_root = ts_state.node_map.get(self.ts_node_id, tree.root_node)

        return CodeChunk.from_ts(
            node=ts_root,
            src=src_file,
            type=self.type,
            tag=self.tag,
        )


class IsolatedCodeChunkModel(BaseModel):
    """
    A code chunk that is isolated from the rest of the codebase.
    """

    type: ChunkType
    uuid: str
    start_line: int
    end_line: int
    start_column: int
    end_column: int
    content: str
    src: Optional[str] = None
    codebase_id: Optional[str] = None

    @classmethod
    def from_chunk(
        cls,
        chunk: CodeChunk,
        include_content_type: Literal[
            "code", "codeblock", "ast_codeblock"
        ] = "codeblock",
    ) -> Self:
        match include_content_type:
            case "code":
                content = chunk.code()
            case "codeblock":
                content = chunk.codeblock()
            case "ast_codeblock":
                content = chunk.ast_codeblock()
        return cls(
            type=chunk.type,
            uuid=str(chunk.uuid),
            start_line=chunk.start_line,
            end_line=chunk.end_line,
            start_column=chunk.start_column,
            end_column=chunk.end_column,
            content=content,
            src=str(chunk.src.path),
            codebase_id=chunk.src.codebase.id,
        )


class FileModel(BaseModel):
    path: str
    file_type: FileType
    lazy: bool = False
    content: Optional[str] = None
    chunks: List[CodeChunkModel] = []

    @classmethod
    def from_file(cls, file: File, include_content: bool = False) -> "FileModel":
        return cls(
            path=str(file.path),
            file_type=file.file_type,
            lazy=file.lazy,
            content=file.content if include_content else None,
            chunks=[
                CodeChunkModel.from_chunk(chunk, include_content)
                for chunk in file.chunks
            ],
        )

    def to_file(self, codebase: Codebase) -> File:
        file = File.new(
            path=self.path, codebase=codebase, file_type=self.file_type, lazy=self.lazy
        )
        if self.content is not None:
            file._content = self.content
        file.chunks = [chunk_model.to_chunk(file) for chunk_model in self.chunks]
        return file


class CodebaseModel(BaseModel):
    id: str
    url: Optional[str] = None
    dir: str
    source_files: Dict[str, FileModel]
    dependency_files: Dict[str, FileModel]
    version: str = "v0.0.1"
    ignore_tests: bool = True

    @classmethod
    def from_codebase(
        cls, codebase: Codebase, include_content: bool = False
    ) -> "CodebaseModel":
        return cls(
            id=codebase.id,
            url=codebase.url,
            dir=str(codebase.dir),
            source_files={
                str(k): FileModel.from_file(v, include_content)
                for k, v in codebase.source_files.items()
            },
            dependency_files={
                str(k): FileModel.from_file(v, include_content)
                for k, v in codebase.dependency_files.items()
            },
            version=codebase.version,
            ignore_tests=codebase.ignore_tests,
        )

    def to_codebase(self) -> Codebase:
        codebase = Codebase(
            id=self.id,
            dir=Path(self.dir),
            source_files={},
            dependency_files={},
            version=self.version,
            ignore_tests=self.ignore_tests,
        )

        # Convert source files
        for path_str, file_model in self.source_files.items():
            codebase.source_files[path_str] = file_model.to_file(codebase)

        # Convert dependency files
        for path_str, file_model in self.dependency_files.items():
            codebase.dependency_files[path_str] = file_model.to_file(codebase)

        # Collect all chunks
        codebase.all_chunks = []
        for file in codebase.source_files.values():
            codebase.all_chunks.extend(file.chunks)

        return codebase
