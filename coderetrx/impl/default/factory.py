from attrs import define
from pathlib import Path
from pydantic import Field

from coderetrx.retrieval.smart_codebase import SmartCodebaseSettings
from coderetrx.utils.embedding import SimilaritySearcher
from coderetrx.static.codebase import Codebase
from .smart_codebase import SmartCodebase

import logging

logger = logging.getLogger(__name__)


@define
class CodebaseFactory:
    @classmethod
    def from_json(cls, data: dict, settings: SmartCodebaseSettings = SmartCodebaseSettings()) -> SmartCodebase:
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
            settings=settings,
        )
        smart_codebase.init_all()
        cls._initialize_similarity_searchers(smart_codebase, settings)
        return smart_codebase

    @classmethod
    def new(cls, id: str, dir: Path, settings: SmartCodebaseSettings = SmartCodebaseSettings()) -> SmartCodebase:
        smart_codebase = SmartCodebase.new(id, dir, settings=settings)
        smart_codebase.init_all()
        cls._initialize_similarity_searchers(smart_codebase, settings)
        return smart_codebase

    @classmethod
    def _initialize_similarity_searchers(
        cls, codebase: SmartCodebase, settings: SmartCodebaseSettings
    ) -> None:
        """
        Initialize SimilaritySearcher instances for the codebase if embeddings are available.

        Args:
            codebase: The codebase to initialize searchers for
        """

        # Initialize symbol name searcher if embeddings are available
        if settings.symbol_name_embedding:
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

        if settings.symbol_content_embedding:
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
        if settings.keyword_embedding:
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
        if settings.symbol_codeline_embedding:
            # Collect all lines from all symbols with metadata
            all_lines = []
            all_metadatas = []
            
            for symbol in codebase.symbols:
                if symbol.chunk:
                    try:
                        logger.debug(
                            f"Collecting lines for symbol {symbol.id}"
                        )
                        lines = symbol.chunk.code().split("\n")
                        lines = [line.strip() for line in lines if line.strip()]  # Remove empty lines
                        lines = [line for line in lines if len(line) > 2]  # Remove lines that are too short
                        lines = list(set(lines))  # Remove duplicates
                        for line in lines:
                            all_lines.append(line)
                            all_metadatas.append({"symbol_id": symbol.id})
                        
                        logger.debug(
                            f"Collected {len(lines)} lines for symbol {symbol.id}"
                        )
                    except Exception as e:
                        logger.error(
                            f"Failed to collect lines for symbol {symbol.id}: {e}"
                        )
            
            # Create single collection for all lines with metadata
            if all_lines:
                try:
                    logger.info(f"Creating unified codeline searcher with {len(all_lines)} total lines")
                    codebase.symbol_codeline_searcher = SimilaritySearcher(
                        f"{codebase.id}_symbol_codelines",
                        all_lines,
                        metadatas=all_metadatas
                    )
                    logger.info("Unified symbol codeline searcher initialized successfully")
                except Exception as e:
                    logger.error(f"Failed to initialize unified symbol codeline searcher: {e}")