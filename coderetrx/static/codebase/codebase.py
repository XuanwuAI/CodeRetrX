import logging
import re
import uuid
from collections import defaultdict
from enum import Enum
from os import PathLike
from pathlib import Path
from textwrap import dedent
from typing import (
    Any,
    Dict,
    Generator,
    Iterable,
    List,
    Literal,
    Optional,
    Self,
    Set,
    Tuple,
    TypeVar,
    Union,
)
from urllib.parse import urlparse
from uuid import UUID

from attrs import Factory, define
from ignore import Walk
from lspyc.handle.protocol import Range, Position, Location
from lspyc.mlclient import MutilLangClient
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


from coderetrx.static.ripgrep import (
    GrepMatchResult,
    ripgrep_search,
    ripgrep_search_symbols,
)

from .languages import (
    OBJLIKE_TAGS,
    IDXSupportedLanguage,
    IDXSupportedTag,
    get_language,
    is_dependency,
    is_sourcecode,
)
from .parsers import CodebaseParser, ParserFactory


class ChunkType(str, Enum):
    PRIMARY = "primary"
    REFERENCE = "reference"
    IMPORT = "import"
    VARIABLE = "variable"
    QUERY_RESULT = "query_result"
    OTHER = "other"


def is_utf8(file_path):
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            f.read()
        return True
    except UnicodeDecodeError:
        return False


def hunk_uuid(
    path: PathLike, start_line: int, end_line: int, start_column: int, end_column: int
):
    return uuid.uuid5(
        uuid.NAMESPACE_OID,
        str(path)
        + str(start_line)
        + str(end_line)
        + str(start_column)
        + str(end_column),
    )


def get_collection_name(url_or_id: str, version: str):
    return f"{url_or_id.replace('/', '_').replace(':', '_')}-{version}"


@define
class CodeHunk:
    """Base class for representing a range of code in a file."""

    start_line: int
    end_line: int
    start_column: int
    end_column: int
    src: "File"
    uuid: UUID

    # @cached(cache=lines_cache)
    def lines(self):
        return self.src.lookup_lines(self)

    @classmethod
    def new(
        cls,
        src: "File",
        start_line: int,
        end_line: int,
        start_column: int,
        end_column: int,
    ):
        return cls(
            src=src,
            start_line=start_line,
            end_line=end_line,
            start_column=start_column,
            end_column=end_column,
            uuid=hunk_uuid(src.path, start_line, end_line, start_column, end_column),
        )

    def get_linerange(self):
        """
        Get inclusive line range
        """
        start, end = self.start_line, self.end_line
        if self.end_column == 0 and end > start:
            end -= 1
        return start, end

    # @cached(cache=code_cache)
    def code(
        self,
        do_dedent: bool = True,
        show_line_numbers: bool = False,
        trunc_headlines: Optional[int] = None,
        zero_based_line_numbers: bool = True,
    ):
        start, end = self.get_linerange()
        if trunc_headlines is not None:
            end = min(end, start + trunc_headlines - 1)
        return self.src.lookup(
            (start, end), do_dedent, show_line_numbers, zero_based_line_numbers
        )

    # @cached(cache=codeblock_cache)
    def codeblock(self, show_line_numbers: bool = False):
        code = self.code()
        if show_line_numbers:
            lines = code.splitlines()
            line_numbers = range(self.start_line, self.start_line + len(lines))
            numbered_lines = [
                f"{num:4d} | {line}" for num, line in zip(line_numbers, lines)
            ]
            code = "\n".join(numbered_lines)
        return f"```{self.src.path}:{self.start_line}:{self.start_column}-{self.end_line}:{self.end_column}\n{code}\n```"

    def lang(self):
        return self.src.lang()

    def includes(self, other: "CodeHunk"):
        return self.start_line <= other.start_line and self.end_line >= other.end_line

    @property
    def id(self):
        return str(self.uuid)

    def __hash__(self):
        return self.uuid.int

    def __repr__(self):
        return self.codeblock()

    def __rich__(self):
        return self.codeblock()

    def __len__(self):
        return len(self.code())


@define
class CodeChunk(CodeHunk):
    """A code chunk representing a range of code in a file."""

    type: ChunkType
    parent: Optional[Self] = None
    tag: Optional[IDXSupportedTag] = None
    name: Optional[str] = None

    def to_json(self, include_content: bool = False):
        from .models import CodeChunkModel

        return CodeChunkModel.from_chunk(self, include_content).model_dump()

    @classmethod
    def from_json(cls, data: dict, src: "File", init_ts: bool = True) -> "CodeChunk":
        """
        Create a CodeChunk instance from a JSON representation.
        """
        from .models import CodeChunkModel

        model = CodeChunkModel.model_validate(data)
        chunk = model.to_chunk(src)
        return chunk

    @classmethod
    def new(
        cls,
        src: "File",
        start_line: int,
        end_line: int,
        start_column: int,
        end_column: int,
        chunk_type: ChunkType,
        tag: Optional[IDXSupportedTag] = None,
        name: Optional[str] = None,
    ):
        """Parser-agnostic factory method for creating code chunks."""
        return cls(
            type=chunk_type,
            tag=tag,
            name=name,
            start_line=start_line,
            end_line=end_line,
            start_column=start_column,
            end_column=end_column,
            src=src,
            uuid=hunk_uuid(src.path, start_line, end_line, start_column, end_column),
        )

    def symbol_name(self) -> Optional[str]:
        if self.name is None:
            return None
        symb_name = self.name
        par = self.parent
        while par:
            if par.name and par.tag in OBJLIKE_TAGS:
                symb_name = f"{par.name}.{symb_name}"
            par = par.parent
        return symb_name

    def ast_codeblock(
        self,
        par_headlines: int = 3,
        show_line_numbers: bool = False,
        show_imports: bool = False,
        trunc_headlines: Optional[int] = None,
        zero_based_line_numbers: bool = True,
    ):
        shift_line_num = 0 if zero_based_line_numbers else 1
        parts = []
        prev_child = self
        parent = self.parent
        while parent:
            # Skip parent processing since we no longer have ts_root
            line_diff = prev_child.start_line - parent.start_line
            if line_diff > 0:
                par_lines = parent.lines()[: min(par_headlines, line_diff)]
                if show_line_numbers:
                    line_numbers = range(
                        parent.start_line, parent.start_line + len(par_lines)
                    )
                    par_lines = [
                        f"{num+shift_line_num:4d} | {line}"
                        for num, line in zip(line_numbers, par_lines)
                    ]
                if line_diff > par_headlines:
                    par_lines.append("... Omitted Lines ...")
                parts.insert(0, "\n".join(par_lines))
            prev_child = parent
            parent = parent.parent
        if show_imports and not self.type == ChunkType.IMPORT:
            import_chunks = [
                chunk
                for chunk in self.src.chunks
                if chunk.type == ChunkType.IMPORT and chunk.parent is None
            ]
            parts.insert(
                0,
                "\n".join(
                    [
                        chunk.code(
                            show_line_numbers=show_line_numbers,
                            zero_based_line_numbers=zero_based_line_numbers,
                        )
                        for chunk in import_chunks
                    ]
                ),
            )
        parts.append("<CODE_CHUNK_IN_INTEREST>")
        code = self.code(
            do_dedent=False,
            show_line_numbers=show_line_numbers,
            trunc_headlines=trunc_headlines,
            zero_based_line_numbers=zero_based_line_numbers,
        )
        parts.append(code)
        parts.append("</CODE_CHUNK_IN_INTEREST>")
        parts.insert(0, f"```{self.src.path}:{self.src.lang()}")
        parts.append("```")
        return "\n".join(parts)

    def __hash__(self):
        return super().__hash__()

    def __repr__(self):
        return super().__repr__()

    def __rich__(self):
        return self.codeblock()


# TreeSitterState moved to parsers/treesitter.py


FileType = Literal["source", "dependency"]


# MARK: File
@define
class File:
    codebase: "Codebase"
    path: PathLike
    file_type: FileType
    lazy: bool = False

    # cache of file content
    _content: str | None = None
    _lines: List[str] | None = None

    chunks: List[CodeChunk] = Factory(list)

    # Parser system
    _parser: Optional[CodebaseParser] = None
    _parse_state: Optional[Any] = None

    @classmethod
    def new(
        cls,
        path: PathLike | str,
        codebase: "Codebase",
        file_type: FileType,
        lazy: bool = False,
    ):
        return cls(path=Path(path), codebase=codebase, file_type=file_type, lazy=lazy)

    def lang(self):
        if self.file_type == "source":
            return get_language(self.path)
        return None

    @property
    def content(self):
        targ_path = Path(self.codebase.dir) / Path(self.path)
        if self.lazy:
            with open(targ_path, "r") as f:
                return f.read()
        if self._content is None:
            with open(targ_path, "r") as f:
                self._content = f.read()
        return self._content

    @property
    def lines(self):
        if self.lazy:
            return self.content.splitlines()
        if self._lines is None:
            self._lines = self.content.splitlines()
        return self._lines

    # Legacy tree-sitter methods removed - now handled by parser system

    def init_parser(self, parser: Optional[CodebaseParser] = None) -> Any:
        """
        Initialize parser for this file.

        Args:
            parser: Parser instance to use. If None, uses codebase parser.

        Returns:
            Parser-specific state
        """
        if self._parser is None:
            self._parser = parser or self.codebase.get_parser()

        if self._parser is None:
            raise ValueError("No parser available for file parsing")

        if self._parse_state is None:
            self._parse_state = self._parser.parse_file(self)

        return self._parse_state

    def get_parser(self) -> Optional[CodebaseParser]:
        """Get the parser instance for this file."""
        return self._parser

    def init_chunks_with_parser(
        self, parser: Optional[CodebaseParser] = None
    ) -> List[CodeChunk]:
        """
        Initialize chunks using the parser system.

        Args:
            parser: Parser instance to use. If None, uses codebase parser.

        Returns:
            List of extracted chunks
        """
        if self.chunks:  # Already initialized
            return self.chunks

        parse_state = self.init_parser(parser)
        if self._parser is None:
            raise ValueError("No parser available")
        self.chunks = self._parser.extract_chunks(self, parse_state)
        return self.chunks

    def init_all(self):
        """Initialize chunks using the codebase parser system."""
        if hasattr(self.codebase, "get_parser") and self.codebase.get_parser():
            self.init_chunks_with_parser()
        else:
            raise ValueError("No parser configured for codebase")

    # @cached(cache=lookup_lines_cache)
    def lookup_lines(
        self,
        x: CodeHunk | Tuple[int, int] | int,
        show_line_numbers: bool = False,
        zero_based_line_numbers: bool = True,
    ) -> List[str]:
        if isinstance(x, CodeHunk):
            start, end = x.get_linerange()
            lines = self.lines[start : end + 1]
        elif isinstance(x, tuple):
            lines = self.lines[x[0] : x[1] + 1]
        elif isinstance(x, int):
            lines = [self.lines[x]]
        else:
            raise ValueError(f"Invalid type: {type(x)}")

        if show_line_numbers:
            if isinstance(x, CodeHunk):
                start_line = x.start_line
            elif isinstance(x, tuple):
                start_line = x[0]
            else:
                start_line = x

            shift_line_num = 0 if zero_based_line_numbers else 1
            line_numbers = range(start_line, start_line + len(lines))
            return [
                f"{num+shift_line_num:4d} | {line}"
                for num, line in zip(line_numbers, lines)
            ]

        return lines

    # @cached(cache=lookup_cache)
    def lookup(
        self,
        x: CodeHunk | Tuple[int, int] | int,
        do_dedent: bool = True,
        show_line_numbers: bool = False,
        zero_based_line_numbers: bool = True,
    ) -> str:
        lines = self.lookup_lines(
            x, show_line_numbers, zero_based_line_numbers=zero_based_line_numbers
        )
        if do_dedent and not show_line_numbers:
            return dedent("\n".join(lines))
        return "\n".join(lines)

    def codeblock(self, show_line_numbers: bool = False):
        lines = self.lines
        if show_line_numbers:
            numbered_lines = [f"{i:4d} | {lines[i]}" for i in range(len(lines))]
            content = dedent("\n".join(numbered_lines))
            return f"```{self.path}\n{content}\n```"
        content = dedent("\n".join(lines))
        return f"```{self.path}\n{content}\n```"

    def to_json(self, include_content: bool = False):
        from .models import FileModel

        return FileModel.from_file(self, include_content).model_dump()

    def grep(
        self,
        patterns: str | List[str],
        languages: List[IDXSupportedLanguage] | None = None,
    ) -> Iterable[CodeHunk]:
        if languages is not None and self.lang() not in languages:
            return []

        if isinstance(patterns, str):
            patterns = [patterns]

        for i, line in enumerate(self.lines):
            for pattern in patterns:
                # Use regex with word boundaries \b to match whole words only
                # Compile the pattern first, then search
                regex_pattern = re.compile(r"\b" + re.escape(pattern) + r"\b")
                if regex_pattern.search(line):
                    yield CodeHunk.new(
                        start_line=i,
                        end_line=i,
                        start_column=0,
                        end_column=len(line),
                        src=self,
                    )
                    # Break after first match to avoid duplicate chunks for the same line
                    break

    def primary_chunks(self) -> List[CodeChunk]:
        return [chunk for chunk in self.chunks if chunk.type == ChunkType.PRIMARY]

    def get_lines(self, max_chars=0):
        chunks = self.primary_chunks()  # Already sorted by start_line
        active_chunks = []  # List of chunks that currently contain the line
        chunk_idx = 0  # Index of next chunk to consider
        results: List[CodelineDocument] = []

        for i, line in enumerate(self.lines):
            # Remove chunks that end before the current line
            active_chunks = [chunk for chunk in active_chunks if chunk.end_line >= i]

            # Add chunks that start at or before the current line and haven't been processed yet
            while chunk_idx < len(chunks) and chunks[chunk_idx].start_line <= i:
                chunk = chunks[chunk_idx]
                # Only add chunks that actually contain the current line
                if chunk.end_line >= i:
                    active_chunks.append(chunk)
                chunk_idx += 1

            # Collect IDs of all chunks that contain the current line
            symbol_ids = [chunk.id for chunk in active_chunks]

            if len(line):
                results.append(
                    CodelineDocument(
                        file_path=str(self.path),
                        start_line=i,
                        content=line,
                        symbol_ids=symbol_ids,
                    )
                )

        cur_buf = []
        cur_symbol_ids = []
        cur_buf_len = 0

        def get_new_doc():
            nonlocal cur_buf, cur_buf_len, cur_symbol_ids
            content = "\n".join([doc.content for doc in cur_buf])
            new_doc = CodelineDocument(
                file_path=str(self.path),
                start_line=cur_buf[0].start_line,
                end_line=cur_buf[-1].start_line,
                content=content,
                symbol_ids=cur_buf[0].symbol_ids,
            )
            cur_buf = []
            cur_buf_len = 0
            return new_doc

        for doc in results:
            if len(cur_buf) and (
                cur_buf_len > max_chars or cur_symbol_ids != doc.symbol_ids
            ):
                yield get_new_doc()
            cur_buf.append(doc)
            cur_buf_len += len(doc.content)
            cur_symbol_ids = doc.symbol_ids
        if len(cur_buf):
            yield get_new_doc()

    @classmethod
    def jit_for_testing(cls, filename: str, content: str) -> Self:
        res = cls(
            path=Path(filename),
            codebase=Codebase.jit_for_testing(),
            file_type="source",
            lazy=False,
        )
        res._content = content
        return res

    @classmethod
    def from_json(
        cls, data: dict, codebase: "Codebase", load_content: bool = False
    ) -> "File":
        """
        Create a File instance from a JSON representation.
        Args:
            data: The JSON data
            load_content: If True, will load the file content from disk
        """
        from .models import FileModel

        model = FileModel.model_validate(data)
        if load_content:
            with open(Path(model.path), "r") as f:
                model.content = f.read()
        return model.to_file(codebase)

    def __hash__(self):
        return hash((self.codebase.id, self.path))

    def __repr__(self):
        return f"File(path='{self.path}', type='{self.file_type}', lazy={self.lazy})"


@define
class Symbol:
    name: str
    type: Literal["function", "class", "dependency"]
    file: File
    chunk: CodeChunk
    id: str = ""
    selection_range: Optional[Range] = None  # LSP identifier range

    def __attrs_post_init__(self):
        if not self.id:
            self.id = self.chunk.id

    def __repr__(self):
        return f"Symbol(name='{self.name}', type='{self.type}', id='{self.id}', file='{self.file.path}')"

    def to_json(self):
        from .models import SymbolModel

        return SymbolModel.from_symbol(self).model_dump()

    @classmethod
    def from_json(cls, data: dict, codebase: "Codebase") -> "Symbol":
        from .models import SymbolModel

        model = SymbolModel.model_validate(data)
        return model.to_symbol(codebase)


@define
class Keyword:
    content: str
    referenced_by: List[File] = Factory(list)

    def __repr__(self):
        return f"Keyword(content='{self.content}', referenced_by_count={len(self.referenced_by)})"

    def to_json(self):
        from .models import KeywordModel

        return KeywordModel.from_keyword(self).model_dump()

    @classmethod
    def from_json(cls, data: dict, codebase: "Codebase") -> "Keyword":
        from .models import KeywordModel

        model = KeywordModel.model_validate(data)
        return model.to_keyword(codebase)


@define
class Dependency:
    id: str
    name: str
    imported_by: List[File] = Factory(list)

    def __repr__(self):
        return f"Dependency(name='{self.name}', id='{self.id}', imported_by_count={len(self.imported_by)})"

    def to_json(self):
        from .models import DependencyModel

        return DependencyModel.from_dependency(self).model_dump()

    @classmethod
    def from_json(cls, data: dict, codebase: "Codebase") -> "Dependency":
        from .models import DependencyModel

        model = DependencyModel.model_validate(data)
        return model.to_dependency(codebase)


@define
class CallGraphEdge:
    file: File
    from_id: str
    from_name: str
    to_id: str
    to_name: str
    line: int
    id: str = ""

    def __repr__(self):
        return f"CallGraphEdge(from='{self.from_name}', to='{self.to_name}', line={self.line}, id='{self.id}')"

    def uuid(self) -> str:
        namespace = uuid.NAMESPACE_OID
        name = str(self.from_id) + str(self.to_id)
        return str(uuid.uuid5(namespace, name))

    def __attrs_post_init__(self):
        if not self.id:
            self.id = self.uuid()

    def to_json(self):
        from .models import CallGraphEdgeModel

        return CallGraphEdgeModel.from_edge(self).model_dump()

    @classmethod
    def from_json(cls, data: dict, codebase: "Codebase") -> "CallGraphEdge":
        from .models import CallGraphEdgeModel

        model = CallGraphEdgeModel.model_validate(data)
        return model.to_edge(codebase)


class CodelineDocument(BaseModel):
    file_path: str
    # Range is inclusive [start_line, end_line]
    start_line: int = Field(description="0-indexed line number of the code line")
    end_line: Optional[int] = Field(
        default=None, description="0-indexed end line number of the code line"
    )
    content: str
    symbol_ids: List[str]


class CodeLine(BaseModel):
    """Pydantic model representing a code line entry with metadata."""

    model_config = {"arbitrary_types_allowed": True}

    line_content: str = Field(description="The content of the code line")
    symbol: Symbol = Field(description="Symbol object containing this line")
    score: float = Field(description="Vector similarity score for this line")

    @classmethod
    def new(cls, line_content: str, symbol: Symbol, score: float = 0.0) -> "CodeLine":
        return cls(line_content=line_content, symbol=symbol, score=score)


CodeElement = Symbol | Keyword | Dependency | File | CallGraphEdge

CodeElementTypeVar = TypeVar("CodeElementTypeVar", bound=CodeElement)


# MARK: Codebase
@define
class Codebase:
    id: str
    dir: PathLike
    source_files: Dict[str, File]
    dependency_files: Dict[str, File]
    all_chunks: List[CodeChunk] = Factory(list)
    symbols: List[Symbol] = Factory(list)
    keywords: List[Keyword] = Factory(list)
    dependencies: List[Dependency] = Factory(list)
    url: Optional[str] = None
    version: str = "v0.0.1"
    ignore_tests: bool = True

    # Parser system
    _parser: Optional[CodebaseParser] = None

    # LSP client for advanced code navigation
    _lsp_client: Optional[Any] = None  # MutilLangClient from lspyc
    _lsp_initialized: bool = False

    _chunks_initialized: bool = False
    _symbols_initialized: bool = False
    _keywords_initialized: bool = False
    _dependencies_initialized: bool = False
    _call_graph_initialized: bool = False

    @classmethod
    def from_json(
        cls, data: dict, languages: Optional[List[IDXSupportedLanguage]] = None
    ) -> "Codebase":
        """
        Create a Codebase instance from a JSON representation.
        """
        from .models import CodebaseModel

        model = CodebaseModel.model_validate(data)
        return model.to_codebase(languages=languages)

    @classmethod
    def new(
        cls,
        id,
        dir: PathLike,
        url: Optional[str] = None,
        lazy: bool = False,
        version: str = "v0.0.1",
        ignore_tests: bool = True,
        languages: Optional[List[IDXSupportedLanguage]] = None,
        parser: Optional[Union[str, CodebaseParser]] = None,
        max_file_size: int | None = None,
        **parser_kwargs,
    ) -> Self:
        dir = Path(dir)
        res = cls(
            id=id,
            dir=dir,
            source_files={},
            dependency_files={},
            version=version,
            url=url,
            ignore_tests=ignore_tests,
        )

        # Initialize parser - default to TreeSitter if none specified
        if parser is not None:
            if isinstance(parser, str):
                res._parser = ParserFactory.get_parser(parser, **parser_kwargs)
            else:
                res._parser = parser
        else:
            # Default to TreeSitter parser
            res._parser = ParserFactory.get_parser("treesitter", **parser_kwargs)
        for entry in Walk(dir):
            path = entry.path()
            if not path.is_file():
                continue
            if max_file_size is not None:
                if path.stat().st_size > max_file_size:
                    continue
            if not is_utf8(path):
                continue
            relative_path = path.relative_to(dir)
            if is_sourcecode(path):
                # Check if file language is in the allowed languages filter
                if languages is not None:
                    file_lang = get_language(path)
                    if file_lang not in languages:
                        continue
                res.source_files[str(relative_path)] = File.new(
                    path=relative_path, codebase=res, file_type="source", lazy=lazy
                )
            if is_dependency(path):
                res.dependency_files[str(relative_path)] = File.new(
                    path=relative_path, codebase=res, file_type="dependency", lazy=lazy
                )
        return res

    def get_parser(self) -> Optional[CodebaseParser]:
        """Get the parser instance for this codebase."""
        return self._parser

    async def init_lsp(self):
        """
        Initialize LSP client for advanced code navigation.

        Must be called before using:
        - LSP parser
        - get_function_calls()
        - get_symbol_definition()
        - get_symbol_references()

        Example:
            >>> codebase = Codebase.new(id="repo", dir="/path")
            >>> await codebase.init_lsp()
            >>> calls = await codebase.get_function_calls(symbol)
        """
        if self._lsp_initialized:
            return
        self._lsp_client = MutilLangClient(str(self.dir))
        self._lsp_initialized = True
        logger.info("LSP client initialized for workspace: %s", self.dir)

    async def close_lsp(self):
        """
        Shutdown LSP client and cleanup resources.

        Should be called in async context when done with LSP functionality.

        Example:
            >>> await codebase.init_lsp()
            >>> # ... use LSP features ...
            >>> await codebase.close_lsp()
        """
        if self._lsp_client:
            try:
                await self._lsp_client.shutdown()
                logger.info("LSP client shutdown successfully")
            except Exception as e:
                logger.warning(f"Failed to shutdown LSP client: {e}")
            finally:
                self._lsp_client = None
                self._lsp_initialized = False

    @property
    def lsp_available(self) -> bool:
        """Check if LSP functionality is available."""
        return self._lsp_initialized and self._lsp_client is not None

    async def get_function_calls(
        self, symbol: Symbol, include_declaration: bool = False
    ) -> List[tuple[str, Range]]:
        """
        Get all locations where a function/method is called.

        Args:
            symbol: The Symbol to find calls for
            include_declaration: Include the definition location

        Returns:
            List of CodeChunk objects representing call sites

        Raises:
            ValueError: If LSP is not initialized

        Example:
            >>> await codebase.init_lsp()
            >>> function_symbol = codebase.symbols[0]
            >>> call_sites = await codebase.get_function_calls(function_symbol)
            >>> for call in call_sites:
            ...     print(f"Called at {call.src.path}:{call.start_line}")
        """
        if not self.lsp_available:
            raise ValueError(
                "LSP not initialized. Call await codebase.init_lsp() first."
            )

        assert self._lsp_client is not None, "LSP client should be available"

        # Get references using LSP
        file_path = symbol.file.path
        select_range = symbol.selection_range
        if select_range is None:
            raise ValueError("Symbol has no selection range")
        line = select_range["start"]["line"]
        col = select_range["start"]["character"]

        res = await self._lsp_client.get_references(
            str(file_path), line, col, include_declaration=include_declaration
        )
        def uri2relative(uri: str) -> str:
            uri = uri.replace("file://", "")
            root_dir = Path(self.dir)
            return str(Path(uri.replace("file://", "")).relative_to(root_dir))
        return [(uri2relative(ref["uri"]), ref["range"]) for ref in res]

    async def get_definition(
        self, file_path: str, position: Position
    ) -> Optional[Symbol]:
        """
        Get the definition location of a symbol (useful for references).

        Args:
            symbol: The Symbol to find definition for

        Returns:
            Symbol at the definition location, or None if not found

        Raises:
            ValueError: If LSP is not initialized

        Example:
            >>> await codebase.init_lsp()
            >>> ref_symbol = codebase.symbols[10]  # Some reference
            >>> definition = await codebase.get_symbol_definition(ref_symbol)
            >>> print(f"Defined at {definition.file.path}:{definition.chunk.start_line}")
        """
        if not self.lsp_available:
            raise ValueError(
                "LSP not initialized. Call await codebase.init_lsp() first."
            )

        assert self._lsp_client is not None, "LSP client should be available"
        definitions = await self._lsp_client.get_definition(
            file_path, position["line"], position["character"]
        )

        if not definitions:
            return None

        # Return the first definition (usually there's only one)
        for loc in definitions:
            uri = loc["uri"]
            root_path = Path(self.dir)
            rel_path = Path(urlparse(uri).path).relative_to(root_path)
            target_file = self.get_file(rel_path)
            for sym in self.symbols:
                if sym.file == target_file and sym.selection_range == loc["range"]:
                    return sym
        return None

    async def get_symbol_references(
        self, symbol: Symbol, include_declaration: bool = False
    ) -> List[tuple[str, Range]]:
        """
        Get all references to a symbol.

        Alias for get_function_calls() with a more general name for non-function symbols.

        Args:
            symbol: The Symbol to find references for
            include_declaration: Include the definition location

        Returns:
            List of CodeChunk objects representing reference locations

        Raises:
            ValueError: If LSP is not initialized
        """
        return await self.get_function_calls(symbol, include_declaration)

    def set_parser(self, parser: Union[str, CodebaseParser], **kwargs):
        """
        Set the parser for this codebase.

        Args:
            parser: Parser instance or parser type string
            **kwargs: Additional parser configuration
        """
        if isinstance(parser, str):
            self._parser = ParserFactory.get_parser(parser, **kwargs)
        else:
            self._parser = parser

        # Reset initialization flags since we're changing the parser
        self._chunks_initialized = False

    def init_chunks(self) -> List[CodeChunk]:
        if self._chunks_initialized:
            return self.all_chunks

        if not self._parser:
            raise ValueError(
                "No parser configured for codebase. Use set_parser() to configure a parser."
            )

        logger.info(f"Using parser: {type(self._parser).__name__}")
        self._parser.init_chunks(self)
        self._chunks_initialized = True
        logger.info(
            f"chunk extraction complete: found {len(self.all_chunks)} chunks across the codebase"
        )
        return self.all_chunks

    def _extract_symbols(self):
        if self._symbols_initialized:
            return self.symbols

        if not self._parser:
            raise ValueError(
                "No parser configured for codebase. Use set_parser() to configure a parser."
            )

        self._parser.init_symbols(self)
        self._symbols_initialized = True
        logger.info(
            f"Symbol extraction complete: found {len(self.symbols)} symbols across the codebase"
        )
        return self.symbols

    def _extract_keywords(self):
        """Extract keywords from source code files."""
        if self._keywords_initialized:
            return self.keywords

        if not self._parser:
            raise ValueError(
                "No parser configured for codebase. Use set_parser() to configure a parser."
            )

        self._parser.init_keywords(self)
        self._keywords_initialized = True
        logger.info(
            f"Keyword extraction complete: found {len(self.keywords)} unique keywords across the codebase"
        )
        return self.keywords

    def _extract_dependencies(self):
        if self._dependencies_initialized:
            return self.dependencies

        if not self._parser:
            raise ValueError(
                "No parser configured for codebase. Use set_parser() to configure a parser."
            )

        self._parser.init_dependencies(self)
        self._dependencies_initialized = True
        logger.info(
            f"Dependency extraction complete: found {len(self.dependencies)} unique dependencies across the codebase"
        )
        return self.dependencies

    def _build_call_graph(self):
        # TODO
        pass

    def init_all(
        self,
        chunks: bool = True,
        symbols: bool = True,
        keywords: bool = True,
        dependencies: bool = True,
        call_graph: bool = True,
    ):
        if chunks:
            self.init_chunks()
        if symbols:
            self._extract_symbols()
        if keywords:
            self._extract_keywords()
        if dependencies:
            self._extract_dependencies()
        if call_graph:
            self._build_call_graph()

    def get_collection_name(self):
        return get_collection_name(self.url or self.id, self.version)

    def to_json(self, include_content: bool = False):
        from .models import CodebaseModel

        return CodebaseModel.from_codebase(self, include_content).model_dump()

    def grep_all(
        self,
        patterns: str | List[str],
        languages: List[IDXSupportedLanguage] | None = None,
    ) -> Dict[str, List[CodeChunk]]:
        res = {}
        if isinstance(patterns, str):
            patterns = [patterns]
        for file in self.source_files.values():
            res[str(file.path)] = list(file.grep(patterns, languages=languages))
        return res

    def primary_chunks(self) -> List[CodeChunk]:
        return [chunk for chunk in self.all_chunks if chunk.type == ChunkType.PRIMARY]

    def get_splited_distinct_chunks(
        self, max_lines: Optional[int] = None
    ) -> List[CodeChunk]:
        """
        Get filtered primary + variable chunks that satisfy the following conditions:
        1. No nested parent-child relationships between any two chunks
        2. Large containers (class/module/interface) exceeding max_lines will be recursively split

        Args:
            max_lines: Maximum line count threshold for container chunks, None means unlimited

        Returns:
            List of filtered chunks
        """
        # Collect all primary and variable chunks
        candidates = [
            chunk
            for chunk in self.all_chunks
            if chunk.type in (ChunkType.PRIMARY, ChunkType.VARIABLE)
        ]

        # Group chunks by file for processing
        chunks_by_file = defaultdict(list)
        for chunk in candidates:
            chunks_by_file[chunk.src.path].append(chunk)

        result = []
        for file_chunks in chunks_by_file.values():
            result.extend(self._split_and_filter_file_chunks(file_chunks, max_lines))
        return result

    def _split_and_filter_file_chunks(
        self, chunks: List[CodeChunk], max_lines: Optional[int]
    ) -> List[CodeChunk]:
        """
        Process chunks from a single file using tree-based algorithm.

        Algorithm:
        1. Build a tree structure from parent-child relationships
        2. Traverse the tree depth-first
        3. For each node, decide whether to split (use children) or keep it
        4. Collect leaf nodes that satisfy the constraints
        """
        if not chunks:
            return []

        # Build children map to represent the tree structure
        children_map: Dict[Optional[CodeChunk], List[CodeChunk]] = defaultdict(list)
        for chunk in chunks:
            children_map[chunk.parent].append(chunk)

        # Sort children by start_line for consistent ordering
        for children in children_map.values():
            children.sort(key=lambda x: x.start_line)

        def traverse_and_collect(chunk: CodeChunk) -> Generator[CodeChunk, None, None]:
            """
            Traverse the tree rooted at chunk and collect valid leaf chunks.

            Returns:
                List of chunks to include in the result (leaf nodes after splitting)
            """
            # Check if this chunk should be split
            if self._should_split_container_chunk(chunk, max_lines):
                children = children_map.get(chunk, [])
                if children:  # Split: recursively process all children
                    for child in children:
                        yield from traverse_and_collect(child)
                else:  # No children to split into, keep the chunk itself
                    yield chunk
            else:  # Don't split, return this chunk as a leaf
                yield chunk

        # Process all top-level chunks (roots of the tree)
        result = []
        top_level_chunks = children_map.get(None, [])
        for chunk in top_level_chunks:
            result.extend(traverse_and_collect(chunk))
        return result

    def _should_split_container_chunk(
        self, chunk: CodeChunk, max_lines: Optional[int]
    ) -> bool:
        """Check if a chunk needs to be split"""
        if max_lines is None:
            return False

        # Only split container types
        if chunk.tag not in [
            "definition.class",
            "definition.module",
            "definition.interface",
        ]:
            return False

        # Calculate line count
        line_count = chunk.end_line - chunk.start_line + 1
        return line_count > max_lines

    async def ripgrep_chunks(
        self,
        queries: List[str],
        symbol_mode: bool = False,
        chunktypes: List[ChunkType] | None = None,
        langs: List[IDXSupportedLanguage] | None = None,
        search_path: PathLike | str | None = None,
        ignore_set: Set[CodeChunk] | None = None,
    ) -> List["GrepChunkResult"]:
        """
        Use ripgrep to search for patterns in the codebase and return matching chunks.

        Args:
            queries: List of regex patterns to search for
            chunktypes: Optional filter for specific chunk types
            langs: Optional filter for specific languages
        Returns:
            List of GrepChunkResult objects containing matching chunks and the patterns they match
        """
        # TODO: support ripgrep glob files by lang -> extension
        logger.debug(
            f"RG: {len(queries)} queries, chunktypes: {chunktypes}, langs: {langs}, search_path: {search_path}"
        )

        # Get ripgrep results
        if symbol_mode:
            results = await ripgrep_search_symbols(
                self.dir,
                queries,
                search_arg=str(search_path) if search_path else None,
            )
        else:
            results = await ripgrep_search(
                self.dir,
                queries,
                search_arg=str(search_path) if search_path else None,
            )
        grouped_by_file: Dict[str, List[GrepMatchResult]] = defaultdict(list)
        for result in results:
            logger.debug(f"RG: {result.file_path} {result.line_number}")
            grouped_by_file[str(result.file_path)].append(result)
        grep_results = []
        for file_path, results in grouped_by_file.items():
            file = self.source_files.get(file_path)
            if file is None:
                continue
            if langs is not None and file.lang() not in langs:
                continue
            # TODO: optimize for perf via sorting and binary search
            for chunk in file.chunks:
                matched_symbols = set()
                for res in results:
                    if ignore_set and chunk in ignore_set:
                        continue
                    if chunktypes is not None and chunk.type not in chunktypes:
                        continue
                    if chunk.start_line <= res.line_number <= chunk.end_line:
                        matched_symbols.update(res.matches)
                if matched_symbols:
                    grep_results.append(
                        GrepChunkResult(chunk=chunk, symbols=matched_symbols)
                    )
        return grep_results

    def get_file(self, path: PathLike | str) -> File | None:
        return self.source_files.get(str(path)) or self.dependency_files.get(str(path))

    def _location_to_chunk(self, location: Dict[str, Any]) -> Optional[CodeChunk]:
        """
        Convert an LSP Location to a CodeChunk.

        Args:
            location: LSP Location with uri and range

        Returns:
            CodeChunk if found, None otherwise
        """

        # Parse URI to get file path
        uri = location["uri"]
        parsed = urlparse(uri)
        file_path = Path(parsed.path)

        # Make relative to workspace
        try:
            rel_path = file_path.relative_to(self.dir)
        except ValueError:
            # File is outside workspace, try as-is
            rel_path = file_path

        # Find the file in codebase
        file = self.get_file(str(rel_path))
        if not file:
            return None

        # Extract position from LSP range
        range_data = location["range"]
        start_line = range_data["start"]["line"]
        start_col = range_data["start"]["character"]
        end_line = range_data["end"]["line"]
        end_col = range_data["end"]["character"]

        # Find existing chunk at this location
        for chunk in file.chunks:
            if (
                chunk.start_line == start_line
                and chunk.start_column == start_col
                and chunk.end_line == end_line
                and chunk.end_column == end_col
            ):
                return chunk

        # Create a new CodeChunk for this location
        # (LSP reference might not correspond to an existing chunk)
        return CodeChunk.new(
            src=file,
            start_line=start_line,
            end_line=end_line,
            start_column=start_col,
            end_column=end_col,
            chunk_type=ChunkType.REFERENCE,
            tag="reference.call",  # Assuming it's a call reference
        )

    def get_all_lines(self, max_chars: int = 0):
        for file in self.source_files.values():
            for line in file.get_lines(max_chars=max_chars):
                yield line

    def cleanup(self):
        """
        Clean up resources held by the codebase, including parser resources.

        Note: This does NOT shutdown the LSP client. Use close_lsp() for that in async context.

        Example:
            >>> codebase.cleanup()  # Sync cleanup for parser
            >>> await codebase.close_lsp()  # Async cleanup for LSP
        """
        if self._parser:
            self._parser.cleanup()

    def __del__(self):
        """
        Ensure cleanup when the object is garbage collected.

        Note: This only cleans up sync resources (parser). LSP client must be
        explicitly closed using close_lsp() in async context before object destruction.
        """
        try:
            self.cleanup()
        except Exception:
            pass  # Ignore errors during cleanup

    @classmethod
    def jit_for_testing(cls) -> Self:
        return cls(
            id="tmp_codebase",
            dir=Path("."),
            source_files={},
            dependency_files={},
        )


@define
class GrepChunkResult:
    chunk: CodeChunk
    symbols: Set[str]


HunkLike = TypeVar("HunkLike", bound=CodeHunk)


def coerce_chunks(
    par_chunks: Iterable[HunkLike], child_chunks: Iterable[HunkLike]
) -> Dict[str, List[HunkLike]]:
    # TODO: optimize for perf
    childs = defaultdict(list)
    for par_chunk in par_chunks:
        for child_chunk in child_chunks:
            if par_chunk.includes(child_chunk):
                childs[par_chunk.id].append(child_chunk)
    return childs


if __name__ == "__main__":
    import sys
    from rich import print

    fp = Path(sys.argv[1])
    if fp.is_file():
        file = File.jit_for_testing(str(fp), open(fp).read())
        file.init_all()
        print("Chunks:")
        for chunk in file.primary_chunks():
            print("-" * 50)
            if chunk.name:
                print("NAME:", chunk.name)
                print("SYMBOL NAME:", chunk.symbol_name())
            print(
                chunk.start_line, chunk.start_column, chunk.end_line, chunk.end_column
            )
            print(chunk.type)
            print(chunk.tag)
            print(chunk.ast_codeblock(show_line_numbers=True))

    if fp.is_dir():
        codebase = Codebase.new(id="tmp_codebase", dir=fp)
        codebase.init_all()
        print("Sample Chunks:")
        for chunk in codebase.primary_chunks()[:100]:
            print("-" * 25 + str(chunk.src.path) + "-" * 25)
            print(chunk.code())
        for sym in codebase.symbols:
            print(sym.name, sym.type)
        for kw in codebase.keywords[:20]:
            print(kw.content, [x.path for x in kw.referenced_by])
        print(f"Number of Chunks: {len(codebase.all_chunks)}")
        print(f"Number of Primary Chunks: {len(codebase.primary_chunks())}")
        print(f"Number of Symbols: {len(codebase.symbols)}")
        print(f"Number of Dependencies: {len(codebase.dependencies)}")
        print(f"Number of Keywords: {len(codebase.keywords)}")
