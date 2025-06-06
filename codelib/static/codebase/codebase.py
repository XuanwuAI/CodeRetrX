import uuid
import os
import re
from collections import defaultdict
from enum import Enum
from os import PathLike
from pathlib import Path
from tqdm import tqdm
from textwrap import dedent
from typing import (
    Dict,
    Iterable,
    List,
    Literal,
    Optional,
    Self,
    Set,
    Tuple,
    TypeVar,
)
from uuid import UUID

from attrs import Factory, define
from ignore import Walk
from tree_sitter import Language as TSLanguage, QueryError
from tree_sitter import Node, Tree
from tree_sitter import Parser as TSParser
from tree_sitter_language_pack import get_language as get_ts_language
from tree_sitter_language_pack import get_parser
import logging

logger = logging.getLogger(__name__)


from .languages import (
    FUNCLIKE_TAGS,
    IMPORT_TAGS,
    OBJLIKE_TAGS,
    PRIMARY_TAGS,
    REFERENCE_TAGS,
    IDXSupportedLanguage,
    IDXSupportedTag,
    get_language,
    get_query,
    is_dependency,
    is_sourcecode,
)
from codelib.static.ripgrep import (
    GrepMatchResult,
    ripgrep_search,
    ripgrep_search_symbols,
)


class ChunkType(str, Enum):
    PRIMARY = "primary"
    REFERENCE = "reference"
    IMPORT = "import"
    QUERY_RESULT = "query_result"
    OTHER = "other"

def is_utf8(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
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

    # @cached(cache=code_cache)
    def code(self, do_dedent: bool = True, show_line_numbers: bool = False):
        return self.src.lookup(self, do_dedent, show_line_numbers)

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
    """A code chunk that is backed by a tree-sitter node."""

    type: ChunkType
    ts_root: Node
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
        if init_ts:
            ts_state = src.ts()
            parsed = ts_state.parser.parse(src.content.encode())
            # Use the node map for efficient lookup
            chunk.ts_root = ts_state.node_map.get(chunk.ts_root.id, parsed.root_node)
        return chunk

    @classmethod
    def from_ts(
        cls,
        node: Node,
        src: "File",
        type: str = "primary",
        tag: IDXSupportedTag | None = None,
        name: str | None = None,
    ):
        return cls(
            type=ChunkType(type),
            tag=tag,
            start_line=node.start_point[0],
            end_line=node.end_point[0],
            start_column=node.start_point[1],
            end_column=node.end_point[1],
            src=src,
            uuid=hunk_uuid(
                src.path,
                node.start_point[0],
                node.end_point[0],
                node.start_point[1],
                node.end_point[1],
            ),
            ts_root=node,
            name=name,
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
    ):
        parts = []
        prev_child = self
        parent = self.parent
        while parent:
            if parent.ts_root.text:
                line_diff = prev_child.start_line - parent.start_line
                par_lines = parent.lines()[: min(par_headlines, line_diff)]
                if show_line_numbers:
                    line_numbers = range(
                        parent.start_line, parent.start_line + len(par_lines)
                    )
                    par_lines = [
                        f"{num:4d} | {line}"
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
                        chunk.code(show_line_numbers=show_line_numbers)
                        for chunk in import_chunks
                    ]
                ),
            )
        parts.append("<CODE_CHUNK_IN_INTEREST>")
        code = self.code(do_dedent=False, show_line_numbers=show_line_numbers)
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


@define
class TreeSitterState:
    parser: TSParser
    ts_language: TSLanguage
    idx_language: IDXSupportedLanguage
    tree: Tree
    node_map: Dict[int, Node] = Factory(dict)

    def build_node_map(self, node: Node):
        """Build a map of node IDs to nodes by walking the tree."""
        self.node_map[node.id] = node
        for child in node.children:
            self.build_node_map(child)


type FileType = Literal["source", "dependency"]


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

    # Tree Sitter
    tree_sitter: Optional[TreeSitterState] = None
    chunks: List[CodeChunk] = Factory(list)

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

    def init_treesitter(self) -> TreeSitterState:
        assert self.file_type == "source"
        lang = get_language(self.path)
        if lang is None:
            raise ValueError(f"Unsupported language for file: {self.path}")
        ts_lang = get_ts_language(lang)
        parser = get_parser(lang)
        tree = parser.parse(self.content.encode())
        ts_state = TreeSitterState(
            parser=parser,
            idx_language=lang,
            ts_language=ts_lang,
            tree=tree,
        )
        # ts_state.build_node_map(tree.root_node) -- perf
        self.tree_sitter = ts_state
        return self.tree_sitter

    def ts(self) -> TreeSitterState:
        """
        Convenience method to get the TreeSitterState for the file.
        """
        assert self.file_type == "source"
        if self.tree_sitter is None:
            return self.init_treesitter()
        else:
            return self.tree_sitter

    def get_block_nodes(self) -> Set[Node]:
        assert self.file_type == "source"
        ts = self.ts()
        try:
            stmt = get_query(ts.idx_language, "tests")
            query = ts.ts_language.query(stmt)
            captures = query.captures(ts.tree.root_node)
            return set(node for nodes in captures.values() for node in nodes)
        except FileNotFoundError:
            logger.debug(f"Test Query file not found for language {ts.idx_language}")
            return set()
        except Exception as e:
            raise e

    def init_chunks(self) -> List[CodeChunk]:
        assert self.file_type == "source"
        ts = self.ts()
        stmt = get_query(ts.idx_language)
        query = ts.ts_language.query(stmt)
        matches = query.matches(ts.tree.root_node)
        block_nodes: Set[Node] = (
            self.get_block_nodes() if self.codebase.ignore_tests else set()
        )

        for match in matches:
            match_g = match[1]
            main_tag = next(k for k in match_g.keys() if not k.startswith("name."))
            if main_tag in PRIMARY_TAGS:
                kind = ChunkType.PRIMARY
            elif main_tag in REFERENCE_TAGS:
                kind = ChunkType.REFERENCE
            elif main_tag in IMPORT_TAGS:
                kind = ChunkType.IMPORT
            else:
                continue

            node = match_g[main_tag][0]
            is_blocked = False
            # TODO: optimize for perf
            for block_node in block_nodes:
                if (
                    block_node.start_point <= node.start_point
                    and block_node.end_point >= node.end_point
                ):
                    is_blocked = True
                    break
            if is_blocked:
                continue

            name: str | None = None
            if f"name.{main_tag}" in match_g:
                _name = match_g[f"name.{main_tag}"][0].text
                if _name is not None:
                    name = _name.decode()

            self.chunks.append(CodeChunk.from_ts(node, self, kind, main_tag, name))
            max_chunk_size: int = int(os.environ.get("MAX_CHUNKS_ONE_FILE", 500))
            if len(self.chunks) > max_chunk_size:
                logger.warning(f"Too many chunks in {self.path}: {len(self.chunks)} > {max_chunk_size}")
                self.chunks = []
                return []

        self.populate_nesting()
        return self.chunks

    def populate_nesting(self):
        # Reference: https://github.com/zed-industries/zed/blob/9bd3e156f5d4db33d1cfc3b5868a493dba8ebc09/crates/language/src/buffer.rs#L3396

        # Sort chunks by start line and then by end line in reverse order
        # This ensures that outer chunks come before inner chunks when they start at the same line
        self.chunks.sort(key=lambda x: (x.start_line, -x.end_line))

        # Stack to keep track of potential parent chunks
        # Each entry is (end_line, chunk) where end_line is the ending line of the chunk
        parent_stack = []

        for chunk in self.chunks:
            # Pop chunks from the stack that end before the current chunk starts
            # These cannot be parents of the current chunk
            while parent_stack and parent_stack[-1][0] < chunk.start_line:
                parent_stack.pop()

            # If there's a chunk on the stack, it contains the current chunk
            # The last chunk on the stack is the immediate parent
            if parent_stack:
                chunk.parent = parent_stack[-1][1]

            # Push the current chunk onto the stack as a potential parent for future chunks
            parent_stack.append((chunk.end_line, chunk))

    def init_all(self):
        self.init_treesitter()
        self.init_chunks()

    # @cached(cache=lookup_lines_cache)
    def lookup_lines(
        self, x: CodeHunk | Tuple[int, int] | int, show_line_numbers: bool = False
    ) -> List[str]:
        if isinstance(x, CodeHunk):
            lines = self.lines[x.start_line : x.end_line + 1]
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

            line_numbers = range(start_line, start_line + len(lines))
            return [f"{num:4d} | {line}" for num, line in zip(line_numbers, lines)]

        return lines

    # @cached(cache=lookup_cache)
    def lookup(
        self,
        x: CodeHunk | Tuple[int, int] | int,
        do_dedent: bool = True,
        show_line_numbers: bool = False,
    ) -> str:
        lines = self.lookup_lines(x, show_line_numbers)
        if do_dedent and not show_line_numbers:
            return dedent("\n".join(lines))
        return "\n".join(lines)

    def codeblock(self, show_line_numbers: bool = False):
        lines = self.lines
        if show_line_numbers:
            numbered_lines = [f"{i:4d} | {lines[i]}" for i in range(len(lines))]
            return f"```{self.path}\n{dedent('\n'.join(numbered_lines))}\n```"
        return f"```{self.path}\n{dedent('\n'.join(lines))}\n```"

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


CodeElement = TypeVar(
    "CodeElement", bound=Symbol | Keyword | Dependency | File | CallGraphEdge
)


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

    _chunks_initialized: bool = False
    _symbols_initialized: bool = False
    _keywords_initialized: bool = False
    _dependencies_initialized: bool = False
    _call_graph_initialized: bool = False

    @classmethod
    def from_json(cls, data: dict) -> "Codebase":
        """
        Create a Codebase instance from a JSON representation.
        """
        from .models import CodebaseModel

        model = CodebaseModel.model_validate(data)
        return model.to_codebase()

    @classmethod
    def new(
        cls,
        id,
        dir: PathLike,
        url: Optional[str] = None,
        lazy: bool = False,
        version: str = "v0.0.1",
        ignore_tests: bool = True,
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
        for entry in Walk(dir):
            path = entry.path()
            if not path.is_file():
                continue
            if not is_utf8(path):
                continue
            relative_path = path.relative_to(dir)
            if is_sourcecode(path):
                res.source_files[str(relative_path)] = File.new(
                    path=relative_path, codebase=res, file_type="source", lazy=lazy
                )
            if is_dependency(path):
                res.dependency_files[str(relative_path)] = File.new(
                    path=relative_path, codebase=res, file_type="dependency", lazy=lazy
                )
        return res

    def init_chunks(self) -> List[CodeChunk]:
        if self._chunks_initialized:
            return self.all_chunks
        logger.info("Starting chunks extraction from source code files...")
        files = list(self.source_files.values())
        for file in tqdm(files):
            chunks = file.init_chunks()
            self.all_chunks.extend(chunks)
        self._chunks_initialized = True
        logger.info(
            f"chunk extraction complete: found {len(self.all_chunks)} chunks across the codebase"
        )
        return self.all_chunks

    def _extract_symbols(self):
        if self._symbols_initialized:
            return self.symbols
        logger.info("Starting symbol extraction from source code files...")
        for chunk in tqdm(self.primary_chunks()):
            if symb_name := chunk.symbol_name():
                if chunk.tag in OBJLIKE_TAGS:
                    type = "class"
                elif chunk.tag in FUNCLIKE_TAGS:
                    type = "function"
                else:
                    continue
                self.symbols.append(
                    Symbol(
                        name=symb_name,
                        file=chunk.src,
                        type=type,
                        id=chunk.id,
                        chunk=chunk,
                    )
                )
        self._symbols_initialized = True
        logger.info(
            f"Symbol extraction complete: found {len(self.symbols)} symbols across the codebase"
        )
        return self.symbols

    def _extract_keywords(self):
        """Extract keywords from source code files."""
        if self._keywords_initialized:
            return self.keywords
        logger.info("Starting keyword extraction from source code files...")

        # Set to store unique keywords
        unique_keywords: dict[str, Keyword] = {}

        # Process each source file
        for file_path, file in tqdm(self.source_files.items()):
            try:
                content = file.content
                words = content.split()
                for word in words:
                    # only reserve words, numbers and '.', '-', '_', '/'
                    normalized_word = re.sub(r"[^a-zA-Z0-9.-_/]", "", word.lower())

                    # Skip words that are too short (less than 3 characters)
                    if len(normalized_word) < 3:
                        continue

                    if normalized_word in unique_keywords:
                        unique_keywords[normalized_word].referenced_by.append(file)
                    else:
                        unique_keywords[normalized_word] = Keyword(
                            content=normalized_word, referenced_by=[file]
                        )

            except Exception as e:
                logger.info(f"Error extracting keywords from file {file_path}: {e}")

        self.keywords = list(unique_keywords.values())
        logger.info(
            f"Keyword extraction complete: found {len(self.keywords)} unique keywords across the codebase"
        )
        self._keywords_initialized = True
        return self.keywords

    def _extract_dependencies(self):
        if self._dependencies_initialized:
            return self.dependencies
        logger.info("Starting dependency extraction from source code files...")
        dependency_symbols: List[Symbol] = []
        for file in tqdm(self.source_files.values()):
            ts = file.ts()
            lang = ts.idx_language
            try:
                pattern = get_query(lang, "fine_imports")
            except FileNotFoundError:
                continue
            query = ts.ts_language.query(pattern)
            captures = query.captures(ts.tree.root_node)
            for capture_name, nodes in captures.items():
                if capture_name != "module":
                    continue
                for node in nodes:
                    raw = node.text.decode() if node.text else ""
                    # strip quotes, angle brackets, etc.
                    module_name = raw.strip("'\"<>")
                    # skip empty or local imports
                    if not module_name or module_name.startswith((".", "/")):
                        continue
                    chunk = CodeChunk.from_ts(node, file, "import", name=module_name)
                    sym = Symbol(
                        name=module_name,
                        type="dependency",
                        file=file,
                        chunk=chunk,
                    )
                    dependency_symbols.append(sym)
                    file.chunks.append(chunk)
        dep_map: Dict[str, List[File]] = defaultdict(list)
        for sym in dependency_symbols:
            dep_map[sym.name].append(sym.file)

        self.dependencies = [
            Dependency(id=sym.id, name=name, imported_by=imported_by)
            for name, imported_by in dep_map.items()
        ]
        self.symbols += dependency_symbols
        self._dependencies_initialized = True
        logger.info(
            f"Dependency extraction complete: found {len(self.dependencies)} unique keywords across the codebase"
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

    async def ripgrep_chunks(
        self,
        queries: List[str],
        symbol_mode: bool = True,
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
