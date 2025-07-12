import asyncio
import re
import subprocess
import time
from pathlib import Path

async def run_command(cmd):
    process = await asyncio.create_subprocess_exec(*cmd, cwd=Path(__file__).parent.parent)
    return await process.wait()

def parse_repos():
    repos = []
    with open("bench/repos.txt", 'r') as f:
        content = f.read()
    
    blocks = [block.strip() for block in content.split('\n\n') if block.strip()]
    for block in blocks:
        lines = block.strip().split('\n')
        if len(lines) >= 2 and lines[1].startswith('https://'):
            repos.append(lines[1].strip())
    
    return repos

async def main():
    repos = parse_repos()
    modes = ["file_name", "symbol_name", "line_per_symbol", "precise", "auto"]
    
    print(f"Running benchmark on {len(repos)} repositories with {len(modes)} modes")

    repo_url = "https://github.com/ollama/ollama"
    for mode in modes:
        print(f"  Running {mode} mode...")
        cmd = ["uv", "run", "scripts/code_retriever.py", "-l", "9", "-f", "--mode", mode, "--repo", repo_url]
        exit_code = await run_command(cmd)
        if exit_code != 0:
            print(f"  Failed with exit code {exit_code}")

    # for i, repo_url in enumerate(repos, 1):
    #     print(f"\n[{i}/{len(repos)}] Processing {repo_url}")
    #
    #     for mode in modes:
    #         print(f"  Running {mode} mode...")
    #         cmd = ["uv", "run", "scripts/code_retriever.py", "-l", "9", "-f", "--mode", mode, "--repo", repo_url, "-t"]
    #         exit_code = await run_command(cmd)
    #         if exit_code != 0:
    #             print(f"  Failed with exit code {exit_code}")
    #
    #     break
    
    print("\nRunning analyze_code_reports...")
    await run_command(["uv", "run", "scripts/analyze_code_reports.py"])
    
    print("Benchmark completed")

if __name__ == "__main__":
    asyncio.run(main())