import os
import json
import logging
from pathlib import Path
from typing import Tuple
from git import Repo, GitCommandError

import logging

logger = logging.getLogger(__name__)


def clone_repo_if_not_exists(repo_url: str, target_path: str) -> str:
    logger.info(f"clone repo: {repo_url}")
    repo_path = Path(target_path)
    if not repo_path.exists():
        try:
            logger.info(f"Cloning repository {repo_url} into {repo_path}")
            # todo: remove ssh clone
            Repo.clone_from(
                repo_url,
                repo_path,
                env=(
                    {"GIT_SSH_COMMAND": f'ssh -i {Path(__file__).parent/"sg_rsa"}'}
                    if repo_url.endswith(".git")
                    else None
                ),
                depth=1,
            )
        except GitCommandError as e:
            repo_path.unlink(missing_ok=True)
            raise Exception(f"Clone failed: {e}")
    return repo_path.as_posix()


def get_repo_id(repo_url: str) -> str:
    if repo_url.startswith("http"):
        repo_id = "_".join(repo_url.split("/")[-2:])
    else:
        # todo: only for backward compatibility, we need to remove this in the future
        repo_id = repo_url.split("/")[-1]
    repo_id = repo_id.replace(".git", "")
    return repo_id

def get_data_dir():
    return Path(__file__).parent.parent.parent / ".data"
