import asyncio
from coderetrx.impl.default import CodebaseFactory, TopicExtractor
from coderetrx.retrieval import coderetrx_filter
from coderetrx.retrieval.code_recall import CodeRecallSettings
from coderetrx.retrieval.smart_codebase import SmartCodebaseSettings
from coderetrx.utils.git import clone_repo_if_not_exists, get_repo_id
from coderetrx.utils.path import get_data_dir

async def main():
    # Set up the repository URL and path
    repo_url = "https://github.com/TecharoHQ/anubis.git"
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
    
    # Set the granularity and coarse recall strategy
    result, llm_output = await coderetrx_filter(
        codebase=codebase,
        subdirs_or_files=["/"],
        prompt="The code snippet contains a function call that dynamically executes code or system commands. Examples include Python's `eval()`, `exec()`, or functions like `os.system()`, `subprocess.run()` (especially with `shell=True`), `subprocess.call()` (with `shell=True`), or `popen()`. The critical feature is that the string representing the code or command to be executed is not a hardcoded literal; instead, it's derived from a variable, function argument, string concatenation/formatting, or an external source such as user input, network request, or LLM output.",
        granularity="symbol_content",
        coarse_recall_strategy="line",
        topic_extractor=topic_extractor,
        settings=settings
    )

    '''
    result, llm_output = await coderetrx_filter(
        codebase=codebase,
        subdirs_or_files=["/"],
        prompt="The code snippet contains a function call that dynamically executes code or system commands. Examples include Python's `eval()`, `exec()`, or functions like `os.system()`, `subprocess.run()` (especially with `shell=True`), `subprocess.call()` (with `shell=True`), or `popen()`. The critical feature is that the string representing the code or command to be executed is not a hardcoded literal; instead, it's derived from a variable, function argument, string concatenation/formatting, or an external source such as user input, network request, or LLM output.",
        granularity="symbol_content",
        coarse_recall_strategy="line",
        topic_extractor=topic_extractor,
        settings=settings
    )
    '''

    print(f"Find {len(result)} results, first {min(5, len(result))} are:")
    for i, location in enumerate(result[:5], 1):
        print(f"  {i}. {location}")

if __name__ == "__main__":
    asyncio.run(main())