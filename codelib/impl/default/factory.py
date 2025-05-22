import os
from attrs import define
from pathlib import Path

from codelib.utils.embedding import SimilaritySearcher
from codelib.static.codebase import Codebase
from .smart_codebase import SmartCodebase

import logging

logger = logging.getLogger(__name__)


@define
class CodebaseFactory:
    @classmethod
    def from_json(cls, data: dict) -> SmartCodebase:
        assert isinstance(data, dict)
        codebase = Codebase.from_json(data)
        smart_codebase = SmartCodebase(
            id=codebase.id,
            dir=codebase.dir,
            source_files=codebase.source_files,
            dependency_files=codebase.dependency_files,
            symbols=codebase.symbols,
            keywords=codebase.keywords,
            dependencies=codebase.dependencies,
        )
        smart_codebase.init_all()
        cls._initialize_similarity_searchers(smart_codebase)
        return smart_codebase

    @classmethod
    def new(cls, id: str, dir: Path) -> SmartCodebase:
        smart_codebase = SmartCodebase.new(id, dir)
        smart_codebase.init_all()
        cls._initialize_similarity_searchers(smart_codebase)
        return smart_codebase

    @classmethod
    def _initialize_similarity_searchers(cls, codebase: SmartCodebase) -> None:
        """
        Initialize SimilaritySearcher instances for the codebase if embeddings are available.

        Args:
            codebase: The codebase to initialize searchers for
        """

        # Initialize symbol name searcher if embeddings are available
        if os.environ.get("SYMBOL_NAME_EMBEDDING"):
            try:
                symbol_names = list(set([symbol.name for symbol in codebase.symbols]))
                logger.info(
                    f"Initializing symbol name similarity searcher with {len(symbol_names)} unique symbols"
                )
                codebase.symbol_name_searcher = SimilaritySearcher(
                    f"{codebase.id}_symbol_names",
                    symbol_names,
                )
                logger.info("Symbol name similarity searcher initialized successfully")
            except Exception as e:
                logger.error(
                    f"Failed to initialize symbol name similarity searcher: {e}"
                )
        else:
            logger.info(
                "Symbol name embeddings feature is not enabled (SYMBOL_NAME_EMBEDDING not set), symbol name searcher not initialized"
            )

        # Initialize symbol content searcher if embeddings are available

        if os.environ.get("SYMBOL_CONTENT_EMBEDDING"):
            try:
                symbol_contents = [
                    symbol.chunk.code() for symbol in codebase.symbols if symbol.chunk
                ]
                logger.info(
                    f"Initializing symbol content similarity searcher with {len(symbol_contents)} symbol contents"
                )
                codebase.symbol_content_searcher = SimilaritySearcher(
                    f"{codebase.id}_symbol_contents",
                    symbol_contents,
                )
                logger.info(
                    "Symbol content similarity searcher initialized successfully"
                )
            except Exception as e:
                logger.error(
                    f"Failed to initialize symbol content similarity searcher: {e}"
                )
        else:
            logger.info(
                "Symbol content embeddings feature is not enabled (SYMBOL_CONTENT_EMBEDDING not set), symbol content searcher not initialized"
            )

        # Initialize keyword searcher if embeddings are available
        if os.environ.get("KEYWORD_EMBEDDING"):
            try:
                keyword_contents = [keyword.content for keyword in codebase.keywords]
                logger.info(
                    f"Initializing keyword similarity searcher with {len(keyword_contents)} unique keywords"
                )
                codebase.keyword_searcher = SimilaritySearcher(
                    f"{codebase.id}_keywords",
                    keyword_contents,
                )
                logger.info("Keyword similarity searcher initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize keyword similarity searcher: {e}")
        else:
            logger.info(
                "Keyword embeddings feature is not enabled (KEYWORD_EMBEDDING not set), keyword searcher not initialized"
            )
