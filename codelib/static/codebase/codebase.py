import uuid
import re
from collections import defaultdict
from enum import Enum
from logging import debug
from os import PathLike
from pathlib import Path
from textwrap import dedent
from typing import Dict, Iterable, List, Literal, Optional, Self, Set, Tuple, TypeVar
from uuid import UUID

from attrs import Factory, define
from ignore import Walk
from tree_sitter import Language as TSLanguage
from tree_sitter import Node, Tree
from tree_sitter import Parser as TSParser
from tree_sitter_language_pack import get_language as get_ts_language
from tree_sitter_language_pack import get_parser

from .languages import (
    IMPORT_TAGS,
    OBJLIKE_TAGS,
    PRIMARY_TAGS,
    REFERENCE_TAGS,
    IDXSupportedLanguage,
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


@define
class CodeChunk(CodeHunk):
    """A code chunk that is backed by a tree-sitter node."""

    type: ChunkType
    ts_root: Node
    parent: Optional[Self] = None
    tag: Optional[str] = None
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
        tag: str | None = None,
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
            debug(f"Test Query file not found for language {ts.idx_language}")
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


@define
class Symbol:
    id: str
    name: str
    type: Literal["function", "class", "dependency"]
    file: File
    chunk: CodeChunk


@define
class Keyword:
    content: str
    referenced_by: List[File] = Factory(list)


@define
class Dependency:
    id: str
    name: str
    imported_by: List[File] = Factory(list)


@define
class CallGraphEdge:
    file: File
    from_id: str
    from_name: str
    to_id: str
    to_name: str
    line: int
    id: str = ""

    def uuid(self) -> str:
        namespace = uuid.NAMESPACE_OID
        name = str(self.from_id) + str(self.to_id)
        return str(uuid.uuid5(namespace, name))

    def __attrs_post_init__(self):
        if not self.id:
            self.id = self.uuid()


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
    ) -> "Codebase":
        dir = Path(dir)
        res = cls(
            id=id,
            dir=dir,
            source_files={},
            dependency_files={},
            version=version,
            url=url,
        )
        for entry in Walk(dir):
            path = entry.path()
            if not path.is_file():
                continue
            rel_path = path.relative_to(dir)
            if is_sourcecode(path):
                res.source_files[str(rel_path)] = File.new(
                    path=rel_path, codebase=res, file_type="source", lazy=lazy
                )
            if is_dependency(path):
                res.dependency_files[str(rel_path)] = File.new(
                    path=rel_path, codebase=res, file_type="dependency", lazy=lazy
                )
        return res

    def init_chunks(self) -> List[CodeChunk]:
        files = list(self.source_files.values())
        for file in files:
            chunks = file.init_chunks()
            self.all_chunks.extend(chunks)
        return self.all_chunks

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
    ):
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
        debug(
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
            debug(f"RG: {result.file_path} {result.line_number}")
            grouped_by_file[str(result.file_path)].append(result)
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
                    yield GrepChunkResult(chunk=chunk, symbols=matched_symbols)

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
        codebase.init_chunks()
        print("Sample Chunks:")
        for chunk in codebase.primary_chunks()[:100]:
            print("-" * 25 + str(chunk.src.path) + "-" * 25)
            print(chunk.code())
        print(f"Number of Chunks: {len(codebase.all_chunks)}")
        print(f"Number of Primary Chunks: {len(codebase.primary_chunks())}")
