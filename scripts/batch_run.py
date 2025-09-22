#!/usr/bin/env python3

import subprocess
import time
from pathlib import Path
from urllib.parse import urlparse


QUERIES = ["vuln", "PQAlgo", "refactor"]


def read_repositories(repos_file: Path) -> list[str]:
    repositories: list[str] = []
    if not repos_file.exists():
        raise FileNotFoundError(f"Repos file not found: {repos_file}")
    for line in repos_file.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("//"):
            continue
        repositories.append(stripped)
    return repositories


def get_repository_name_from_url(url: str) -> str:
    parsed = urlparse(url)
    name = Path(parsed.path).name
    if name.endswith(".git"):
        name = name[:-4]
    return name


def main():
    root_dir = Path(__file__).resolve().parent.parent
    print(root_dir)
    repos_file = root_dir / "bench" / "repos.txt"

    repositories = read_repositories(repos_file)

    # Build all jobs
    jobs: list[tuple[list[str], Path]] = []
    for repo_url in repositories:
        repository_name = get_repository_name_from_url(repo_url)
        for query_name in QUERIES:
            log_file = root_dir / f"retriever_{repository_name}_{query_name}.log"
            cmd = [
                "uv",
                "run",
                "scripts/code_retriever.py",
                "-f",
                "-m",
                "precise",
                "--repo",
                repo_url,
                "-q",
                query_name,
            ]
            jobs.append((cmd, log_file))

    # Run with a cap of 6 concurrent processes
    MAX_PARALLEL = 6
    running: list[tuple[subprocess.Popen, any]] = []

    def prune_finished():
        nonlocal running
        still: list[tuple[subprocess.Popen, any]] = []
        for proc, handle in running:
            if proc.poll() is None:
                still.append((proc, handle))
            else:
                try:
                    handle.close()
                except Exception:
                    pass
        running = still

    for cmd, log_file in jobs:
        while len(running) >= MAX_PARALLEL:
            prune_finished()
            if len(running) >= MAX_PARALLEL:
                time.sleep(1.0)

        print("Launching:", " ".join(cmd), "| log:", str(log_file))
        log_handle = open(log_file, "w", encoding="utf-8")
        proc = subprocess.Popen(
            cmd,
            cwd=root_dir,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        running.append((proc, log_handle))

    while running:
        prune_finished()
        if running:
            time.sleep(1.0)

    print("All jobs completed. Check retriever_*.log for results.")


if __name__ == "__main__":
    main()


