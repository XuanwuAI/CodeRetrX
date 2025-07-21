from typing import Optional
from attrs import define
from collections import defaultdict
from pathlib import Path
from pydantic import Field

from coderetrx.retrieval.smart_codebase import SmartCodebaseSettings
from coderetrx.utils.llm import LLMSettings
from coderetrx.utils.embedding import get_similarity_searcher
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
    def new(cls, id: str, dir: Path, settings: Optional[SmartCodebaseSettings] = None, llm_settings: Optional[LLMSettings] = None) -> SmartCodebase:
        settings = settings or SmartCodebaseSettings()
        smart_codebase = SmartCodebase.new(id, dir, settings=settings, llm_settings=llm_settings)
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
                codebase.symbol_name_searcher = get_similarity_searcher(
                    provider=settings.vector_db_provider,
                    name=f"{codebase.id}_symbol_names",
                    texts=symbol_names,
                )
                logger.info("Symbol name similarity searcher initialized successfully")
            except Exception as e:
                logger.fatal(
                    f"Failed to initialize symbol name similarity searcher: {repr(e)}"
                )
                raise e
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
                codebase.symbol_content_searcher = get_similarity_searcher(
                    provider=settings.vector_db_provider,
                    name=f"{codebase.id}_symbol_contents",
                    texts=symbol_contents,
                )
                logger.info(
                    "Symbol content similarity searcher initialized successfully"
                )
            except Exception as e:
                logger.fatal(
                    f"Failed to initialize symbol content similarity searcher: {e}"
                )
                raise e
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
                codebase.keyword_searcher = get_similarity_searcher(
                    provider=settings.vector_db_provider,
                    name=f"{codebase.id}_keywords",
                    texts=keyword_contents,
                )
                logger.info("Keyword similarity searcher initialized successfully")
            except Exception as e:
                logger.fatal(f"Failed to initialize keyword similarity searcher: {repr(e)}")
        else:
            logger.info(
                "Keyword embeddings feature is not enabled (KEYWORD_EMBEDDING not set), keyword searcher not initialized"
            )
        if settings.symbol_codeline_embedding:
            # Collect all lines from all symbols with metadata
            all_lines = []
            all_metadatas = []
            for line in codebase.get_all_lines():
                all_lines.append(line.content)
                all_metadatas.append(line.model_dump(mode="json"))
            
            # Create single collection for all lines with metadata
            if all_lines:
                try:
                    logger.info(f"Creating unified codeline searcher with {len(all_lines)} total lines")
                    codebase.codeline_searcher = get_similarity_searcher(
                        provider=settings.vector_db_provider,
                        name=f"{codebase.id}_symbol_codelines",
                        texts=all_lines,
                        metadatas=all_metadatas
                    )
                    logger.info("Unified symbol codeline searcher initialized successfully")
                except Exception as e:
                    logger.fatal(f"Failed to initialize unified symbol codeline searcher: {repr(e)}")
                    raise e
