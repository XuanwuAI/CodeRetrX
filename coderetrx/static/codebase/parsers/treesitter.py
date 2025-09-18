from typing import List, Any, Set, Dict, Optional, TYPE_CHECKING
import logging
from collections import defaultdict
from pathlib import Path
from tree_sitter import Language as TSLanguage, Node, Tree
from tree_sitter import Parser as TSParser
from tree_sitter_language_pack import get_language as get_ts_language, get_parser
from attrs import Factory, define
from tqdm import tqdm

from .base import CodebaseParser, ParseError, UnsupportedLanguageError
from ..languages import (
    IDXSupportedLanguage,
    IDXSupportedTag,
    PRIMARY_TAGS,
    REFERENCE_TAGS,
    IMPORT_TAGS,
    OBJLIKE_TAGS,
    FUNCLIKE_TAGS,
    get_language,
    get_query,
)

if TYPE_CHECKING:
    from ..codebase import (
        File,
        CodeChunk,
        ChunkType,
        Codebase,
        Symbol,
        Keyword,
        Dependency,
    )

logger = logging.getLogger(__name__)


@define
class TreeSitterState:
    """Tree-sitter parsing state for a file."""

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


class TreeSitterParser(CodebaseParser):
    """
    Tree-sitter based parser implementation.

    This parser uses tree-sitter to parse source code files and extract
    code chunks using language-specific queries.
    """

    def supports_language(self, language: IDXSupportedLanguage) -> bool:
        """Check if tree-sitter supports the given language."""
        try:
            get_ts_language(language)
            return True
        except Exception:
            return False

    def parse_file(self, file: "File") -> "TreeSitterState":
        """
        Parse a file using tree-sitter.

        Args:
            file: The file to parse

        Returns:
            TreeSitterState object containing parsed tree and metadata

        Raises:
            UnsupportedLanguageError: If the file language is not supported
            ParseError: If parsing fails
        """
        if file.file_type != "source":
            raise ParseError(f"Cannot parse non-source file: {file.path}")

        lang = get_language(file.path)
        if lang is None:
            raise UnsupportedLanguageError(
                f"Unsupported language for file: {file.path}"
            )

        if not self.supports_language(lang):
            raise UnsupportedLanguageError(
                f"Tree-sitter does not support language: {lang}"
            )

        try:
            ts_lang = get_ts_language(lang)
            parser = get_parser(lang)
            tree = parser.parse(file.content.encode())

            ts_state = TreeSitterState(
                parser=parser,
                idx_language=lang,
                ts_language=ts_lang,
                tree=tree,
            )
            return ts_state

        except Exception as e:
            raise ParseError(
                f"Failed to parse file {file.path}: {str(e)}",
                file_path=Path(file.path) if file.path else None,
                original_error=e,
            )

    def extract_chunks(
        self, file: "File", parse_state: "TreeSitterState"
    ) -> List["CodeChunk"]:
        """
        Extract code chunks from a tree-sitter parsed file.

        Args:
            file: The file to extract chunks from
            parse_state: TreeSitterState from parse_file

        Returns:
            List of CodeChunk objects with consistent UUIDs
        """
        from ..codebase import CodeChunk, ChunkType

        chunks = []
        ts = parse_state

        try:
            stmt = get_query(ts.idx_language)
            query = ts.ts_language.query(stmt)
            matches = query.matches(ts.tree.root_node)

            # Get test block nodes if test filtering is enabled
            block_nodes: Set[Node] = (
                self._get_block_nodes(file, ts) if file.codebase.ignore_tests else set()
            )

            # Process each query match
            for match in matches:
                match_g = match[1]
                main_tag = next(k for k in match_g.keys() if not k.startswith("name."))

                # Determine chunk type based on tag
                if main_tag in PRIMARY_TAGS:
                    kind = ChunkType.PRIMARY
                elif main_tag in REFERENCE_TAGS:
                    kind = ChunkType.REFERENCE
                elif main_tag in IMPORT_TAGS:
                    kind = ChunkType.IMPORT
                else:
                    continue

                node = match_g[main_tag][0]

                # Check if node is blocked by test blocks
                if self._is_node_blocked(node, block_nodes):
                    continue

                # Extract symbol name if available
                name: str | None = None
                if f"name.{main_tag}" in match_g:
                    _name = match_g[f"name.{main_tag}"][0].text
                    if _name is not None:
                        name = _name.decode()

                # Create chunk using the new parser-specific method
                chunk = self.create_chunk_from_node(node, file, kind, main_tag, name)
                chunks.append(chunk)

                # Check chunk limit to prevent memory issues
                from coderetrx.retrieval.smart_codebase import SmartCodebaseSettings

                settings = SmartCodebaseSettings()
                max_chunk_size: int = settings.max_chunks_one_file
                if max_chunk_size > 0 and len(chunks) > max_chunk_size:
                    logger.warning(
                        f"Too many chunks in {file.path}: {len(chunks)} > {max_chunk_size}"
                    )
                    return []

        except Exception as e:
            logger.error(f"Failed to extract chunks from {file.path}: {str(e)}")
            return []

        # Populate parent-child relationships
        self._populate_nesting(chunks)
        return chunks

    def _get_block_nodes(self, file: "File", ts: "TreeSitterState") -> Set[Node]:
        """Get test block nodes for filtering."""
        try:
            stmt = get_query(ts.idx_language, "tests")
            query = ts.ts_language.query(stmt)
            captures = query.captures(ts.tree.root_node)
            return set(node for nodes in captures.values() for node in nodes)
        except FileNotFoundError:
            logger.debug(f"Test Query file not found for language {ts.idx_language}")
            return set()
        except Exception as e:
            logger.debug(f"Error getting block nodes: {e}")
            return set()

    def _is_node_blocked(self, node: Node, block_nodes: Set[Node]) -> bool:
        """Check if a node is blocked by test blocks."""
        for block_node in block_nodes:
            if (
                block_node.start_point <= node.start_point
                and block_node.end_point >= node.end_point
            ):
                return True
        return False

    def _populate_nesting(self, chunks: List["CodeChunk"]):
        """
        Populate parent-child relationships between chunks.

        This implements the same nesting logic as the original codebase.
        """
        # Sort chunks by start line and then by end line in reverse order
        chunks.sort(key=lambda x: (x.start_line, -x.end_line))

        # Stack to keep track of potential parent chunks
        parent_stack = []

        for chunk in chunks:
            # Pop chunks from the stack that end before the current chunk starts
            while parent_stack and parent_stack[-1][0] < chunk.start_line:
                parent_stack.pop()

            # If there's a chunk on the stack, it contains the current chunk
            if parent_stack:
                chunk.parent = parent_stack[-1][1]

            # Push the current chunk onto the stack as a potential parent
            parent_stack.append((chunk.end_line, chunk))

    # New interface methods for direct codebase population
    def init_chunks(self, codebase: "Codebase") -> None:
        """
        Parse codebase and directly populate chunks into File objects.

        This method processes each file individually (TreeSitter's natural mode)
        and populates chunks directly into the file objects.
        """
        from tqdm import tqdm

        for file in tqdm(
            codebase.source_files.values(), desc="Initializing chunks with TreeSitter"
        ):
            if file.file_type != "source":
                continue

            try:
                # Parse individual file
                parse_state = self.parse_file(file)

                # Extract chunks and populate directly into file
                chunks = self.extract_chunks(file, parse_state)
                file.chunks.extend(chunks)
                codebase.all_chunks.extend(chunks)

            except Exception as e:
                logger.warning(f"Failed to parse {file.path}: {e}")
                continue

    def init_symbols(self, codebase: "Codebase") -> None:
        """
        Create symbols from existing chunks and populate into codebase.

        Ensures chunks are populated first, then creates Symbol objects.
        """
        # Ensure chunks are populated
        if not codebase.all_chunks:
            self.init_chunks(codebase)

        from ..codebase import Symbol, ChunkType

        # Create Symbol objects from chunks
        for chunk in codebase.all_chunks:
            symbol_name = chunk.symbol_name()
            if chunk.type == ChunkType.PRIMARY and symbol_name:
                symbol_type = "class" if chunk.tag in OBJLIKE_TAGS else "function"
                symbol = Symbol(
                    name=symbol_name,
                    type=symbol_type,
                    file=chunk.src,
                    chunk=chunk,
                    id=chunk.id,
                )
                codebase.symbols.append(symbol)

    def init_dependencies(self, codebase: "Codebase") -> None:
        """
        Extract dependencies using tree-sitter queries and populate into codebase.
        """
        from ..codebase import CodeChunk, Symbol, Dependency, ChunkType

        dependency_symbols: List[Symbol] = []

        for file in tqdm(
            codebase.source_files.values(), desc="Extracting dependencies"
        ):
            if file.file_type != "source":
                continue

            try:
                # Parse file if not already done
                ts = self.parse_file(file)
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
                        chunk = self.create_chunk_from_node(
                            node, file, ChunkType.IMPORT, name=module_name
                        )
                        sym = Symbol(
                            name=module_name,
                            type="dependency",
                            file=file,
                            chunk=chunk,
                        )
                        dependency_symbols.append(sym)
                        file.chunks.append(chunk)

            except Exception as e:
                logger.warning(f"Failed to extract dependencies from {file.path}: {e}")
                continue

        # Group dependencies by module name
        dep_map: Dict[str, List["File"]] = defaultdict(list)
        for sym in dependency_symbols:
            dep_map[sym.name].append(sym.file)

        # Create Dependency objects
        for name, imported_by in dep_map.items():
            dependency = Dependency(
                id=str(hash(name)),  # Simple ID generation
                name=name,
                imported_by=imported_by,
            )
            codebase.dependencies.append(dependency)

        # Add dependency symbols to codebase symbols
        codebase.symbols.extend(dependency_symbols)

    @classmethod
    def create_chunk_from_node(
        cls,
        node: Node,
        file: "File",
        chunk_type: "ChunkType",
        tag: Optional[IDXSupportedTag] = None,
        name: Optional[str] = None,
    ) -> "CodeChunk":
        """
        Create a code chunk from a tree-sitter node.

        This is the tree-sitter specific factory method that replaces from_ts.
        """
        from ..codebase import CodeChunk, ChunkType

        return CodeChunk.new(
            tag=tag,
            name=name,
            start_line=node.start_point[0],
            end_line=node.end_point[0],
            start_column=node.start_point[1],
            end_column=node.end_point[1],
            chunk_type=chunk_type,
            src=file,
        )
