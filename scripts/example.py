import asyncio
from coderetrx.impl.default import CodebaseFactory, TopicExtractor
from coderetrx.retrieval.code_recall import CodeRecallSettings, _multi_strategy_code_recall
from coderetrx.retrieval.smart_codebase import SmartCodebaseSettings
from coderetrx.retrieval.strategy.base import RecallStrategy
from coderetrx.retrieval.strategy.filter_imports_by_vector_and_llm import FilterImportsByVectorAndLLMStrategy
from coderetrx.utils.git import clone_repo_if_not_exists, get_repo_id
from coderetrx.utils.path import get_data_dir
import logging

logging.basicConfig(
    level=logging.INFO,
    format="\033[1;36m%(levelname)s\033[0m \033[1;33m%(pathname)s:%(lineno)d\033[0m: %(message)s"
)

async def main():
    # Set up the repository URL and path
    repo_url = "https://github.com/JettChenT/minisignal"
    repo_path = get_data_dir() / "repos" / get_repo_id(repo_url)

    # Clone the repository if it does not exist
    clone_repo_if_not_exists(repo_url, str(repo_path))
    
    # Create codebase settings with symbol_codeline_embedding enabled
    codebase_settings = SmartCodebaseSettings()
    codebase_settings.symbol_codeline_embedding = True
    
    # Create a codebase instance
    codebase = CodebaseFactory.new(get_repo_id(repo_url), repo_path, settings=codebase_settings)
    
    # Create a topic extractor instance
    topic_extractor = TopicExtractor()
    
    # Initialize code recall settings
    settings = CodeRecallSettings()
    
    # Set the target_type and coarse recall strategy
    result, llm_output = await _multi_strategy_code_recall(
        codebase=codebase,
        subdirs_or_files=["/"],
        prompt="an import statement that introduces cryptographic modules or functions to the file",
        target_type="import",
        coarse_recall_strategy="custom",
        topic_extractor=topic_extractor,
        custom_strategies=[RecallStrategy.FILTER_IMPORTS_BY_VECTOR_AND_LLM],
        settings=settings,
        extend_coarse_recall_element_to_file=False,
        llm_method=codebase.llm_filter
    )

    '''
    result, llm_output = await coderetrx_filter(
        codebase=codebase,
        subdirs_or_files=["/"],
        prompt="The code snippet contains a function call that dynamically executes code or system commands. Examples include Python's `eval()`, `exec()`, or functions like `os.system()`, `subprocess.run()` (especially with `shell=True`), `subprocess.call()` (with `shell=True`), or `popen()`. The critical feature is that the string representing the code or command to be executed is not a hardcoded literal; instead, it's derived from a variable, function argument, string concatenation/formatting, or an external source such as user input, network request, or LLM output.",
        target_type="symbol_content",
        coarse_recall_strategy="line_per_symbol",
        topic_extractor=topic_extractor,
        settings=settings
    )
    '''

    print(f"Find {len(result)} results, first {min(5, len(result))} are:")
    for i, location in enumerate(result[:5], 1):
        print(location.codeblock())

if __name__ == "__main__":
    asyncio.run(main())