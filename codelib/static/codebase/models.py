from pathlib import Path
from typing import Dict, List, Literal, Optional, Self, Union

from pydantic import BaseModel, Field
from logging import warn

from .codebase import (
    ChunkType,
    Codebase,
    CodeChunk,
    CodeHunk,
    File,
    hunk_uuid,
    IDXSupportedTag,
    Symbol,
    Keyword,
    Dependency,
    CallGraphEdge,
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


class SymbolModel(BaseModel):
    name: str
    type: Literal["function", "class", "dependency"]
    file_path: str
    chunk_id: str
    id: str

    @classmethod
    def from_symbol(cls, symbol: Symbol) -> "SymbolModel":
        return cls(
            name=symbol.name,
            type=symbol.type,
            file_path=str(symbol.file.path),
            chunk_id=symbol.chunk.id,
            id=symbol.id,
        )

    def to_symbol(self, codebase: Codebase) -> Symbol:
        file = codebase.get_file(self.file_path)
        if file is None:
            raise ValueError(f"File not found: {self.file_path}")
        chunk = next((c for c in file.chunks if c.id == self.chunk_id), None)
        if chunk is None:
            raise ValueError(f"Chunk not found: {self.chunk_id}")
        return Symbol(
            name=self.name,
            type=self.type,
            file=file,
            chunk=chunk,
            id=self.id,
        )


class KeywordModel(BaseModel):
    content: str
    referenced_by: List[str]  # List of file paths

    @classmethod
    def from_keyword(cls, keyword: Keyword) -> "KeywordModel":
        return cls(
            content=keyword.content,
            referenced_by=[str(f.path) for f in keyword.referenced_by],
        )

    def to_keyword(self, codebase: Codebase) -> Keyword:
        files = []
        for path in self.referenced_by:
            file = codebase.get_file(path)
            if file is None:
                raise ValueError(f"File not found: {path}")
            files.append(file)
        return Keyword(
            content=self.content,
            referenced_by=files,
        )


class DependencyModel(BaseModel):
    id: str
    name: str
    imported_by: List[str]  # List of file paths

    @classmethod
    def from_dependency(cls, dependency: Dependency) -> "DependencyModel":
        return cls(
            id=dependency.id,
            name=dependency.name,
            imported_by=[str(f.path) for f in dependency.imported_by],
        )

    def to_dependency(self, codebase: Codebase) -> Dependency:
        files = []
        for path in self.imported_by:
            file = codebase.get_file(path)
            if file is None:
                raise ValueError(f"File not found: {path}")
            files.append(file)
        return Dependency(
            id=self.id,
            name=self.name,
            imported_by=files,
        )


class CallGraphEdgeModel(BaseModel):
    file_path: str
    from_id: str
    from_name: str
    to_id: str
    to_name: str
    line: int
    id: str

    @classmethod
    def from_edge(cls, edge: CallGraphEdge) -> "CallGraphEdgeModel":
        return cls(
            file_path=str(edge.file.path),
            from_id=edge.from_id,
            from_name=edge.from_name,
            to_id=edge.to_id,
            to_name=edge.to_name,
            line=edge.line,
            id=edge.id,
        )

    def to_edge(self, codebase: Codebase) -> CallGraphEdge:
        file = codebase.get_file(self.file_path)
        if file is None:
            raise ValueError(f"File not found: {self.file_path}")
        return CallGraphEdge(
            file=file,
            from_id=self.from_id,
            from_name=self.from_name,
            to_id=self.to_id,
            to_name=self.to_name,
            line=self.line,
            id=self.id,
        )


class CodebaseModel(BaseModel):
    id: str
    url: Optional[str] = None
    dir: str
    source_files: Dict[str, FileModel]
    dependency_files: Dict[str, FileModel]
    version: str = "v0.0.1"
    ignore_tests: bool = True
    symbols: List[SymbolModel] = []
    keywords: List[KeywordModel] = []
    dependencies: List[DependencyModel] = []
    call_graph_edges: List[CallGraphEdgeModel] = []

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
            symbols=[SymbolModel.from_symbol(s) for s in codebase.symbols],
            keywords=[KeywordModel.from_keyword(k) for k in codebase.keywords],
            dependencies=[
                DependencyModel.from_dependency(d) for d in codebase.dependencies
            ],
            call_graph_edges=[],  # TODO: Implement when call graph is built
        )

    def to_codebase(self) -> Codebase:
        # NOTE: We rely on the fact that the codebase initialization process is deterministic
        codebase = Codebase.new(
            id=self.id,
            dir=Path(self.dir),
            version=self.version,
            ignore_tests=self.ignore_tests,
        )
        has_symbols, has_keywords, has_dependencies = (
            bool(len(self.symbols)),
            bool(len(self.keywords)),
            bool(len(self.dependencies)),
        )

        all_chunks = []
        source_files = {}
        for source_file in self.source_files:
            file_model = self.source_files[source_file]
            file = file_model.to_file(codebase)
            source_files[source_file] = file
            all_chunks.extend(file.chunks)
        codebase.all_chunks = all_chunks 
        codebase._chunks_initialized = True

        #todo: enable callgraph properly
        if has_symbols:
            codebase.symbols = [s.to_symbol(codebase) for s in self.symbols]
            codebase._symbols_initialized = True
        if has_keywords:
            codebase.keywords = [k.to_keyword(codebase) for k in self.keywords]
            codebase._keywords_initialized = True
        if has_dependencies:
            codebase.dependencies = [d.to_dependency(codebase) for d in self.dependencies]
            codebase._dependencies_initialized = True
        

        codebase.init_all(False, not has_symbols, not has_keywords, not has_dependencies, False)
        return codebase
