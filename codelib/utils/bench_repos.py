from pydantic import BaseModel
from codelib.static import Codebase
from codelib.utils.stats import CodebaseStats
import os
import subprocess
import json
import tomllib
import tomli_w
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any
from urllib.parse import urlparse

class StatsDigest(BaseModel):
    num_files: int
    num_lines: int
    num_tokens: int
    primary_language: str

    @classmethod
    def from_codebase_stats(cls, stats: CodebaseStats) -> "StatsDigest":
        return cls(
            num_files=stats.num_files,
            num_lines=stats.num_lines,
            num_tokens=stats.num_tokens,
            primary_language=stats.primary_language
        )

class GitBenchRepo(BaseModel):
    url: str
    commit: str
    stats: StatsDigest


class LockFile(BaseModel):
    """The lockfile data structure containing GitBenchRepo information."""
    repositories: Dict[str, GitBenchRepo] = {}

    @classmethod
    def load(cls, lock_path: Path) -> "LockFile":
        """Load and validate lockfile from TOML file."""
        if not lock_path.exists():
            return cls()
        
        try:
            with open(lock_path, 'rb') as f:
                data = tomllib.load(f)
            return cls.model_validate(data)
        except Exception as e:
            print(f"Error loading lock file {lock_path}: {e}")
            return cls()

    def save(self, lock_path: Path) -> None:
        """Save lockfile to TOML format."""
        try:
            with open(lock_path, 'wb') as f:
                tomli_w.dump(self.model_dump(), f)
            print(f"Saved lock file: {lock_path}")
        except Exception as e:
            print(f"Error saving lock file {lock_path}: {e}")

    def get_repo(self, repo_id: str) -> Optional[GitBenchRepo]:
        """Get repository data by ID."""
        return self.repositories.get(repo_id)

    def set_repo(self, repo_id: str, git_bench_repo: GitBenchRepo) -> None:
        """Set repository data."""
        self.repositories[repo_id] = git_bench_repo


def get_commit_hash(repo_path: Path) -> Optional[str]:
    """
    Get the current commit hash for a repository.

    Args:
        repo_path: Path to the repository

    Returns:
        The commit hash as a string, or None if an error occurred
    """
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            cwd=str(repo_path),
        )
        return result.stdout.strip()
    except subprocess.CalledProcessError as e:
        print(f"Error getting commit hash: {e.stderr}")
        return None


def checkout_commit(repo_path: Path, commit_hash: str) -> bool:
    """
    Checkout a specific commit in a repository.

    Args:
        repo_path: Path to the repository
        commit_hash: The commit hash to checkout

    Returns:
        True if checkout was successful, False otherwise
    """
    try:
        # First try direct checkout
        result = subprocess.run(
            ["git", "checkout", commit_hash],
            check=True,
            capture_output=True,
            text=True,
            cwd=str(repo_path),
        )
        return True
    except subprocess.CalledProcessError as e:
        print(f"Initial checkout of commit {commit_hash} failed: {e.stderr}")

        # Try fetching and pulling
        try:
            print(f"Attempting fetch for repository at {repo_path}...")
            subprocess.run(
                ["git", "fetch", "--all"],
                check=True,
                capture_output=True,
                text=True,
                cwd=str(repo_path),
            )

            print(f"Attempting pull for repository at {repo_path}...")
            subprocess.run(
                ["git", "pull"],
                check=True,
                capture_output=True,
                text=True,
                cwd=str(repo_path),
            )

            # Try checkout again after fetch/pull
            print(f"Retrying checkout of commit {commit_hash}...")
            subprocess.run(
                ["git", "checkout", commit_hash],
                check=True,
                capture_output=True,
                text=True,
                cwd=str(repo_path),
            )
            return True
        except subprocess.CalledProcessError as fetch_error:
            print(
                f"Error trying to fetch/pull and checkout commit {commit_hash}: {fetch_error.stderr}"
            )
            return False


def get_repo_id(repo_url: str) -> str:
    """Extract a unique identifier from a repository URL."""
    parsed = urlparse(repo_url)
    path_parts = parsed.path.strip('/').split('/')
    if len(path_parts) >= 2:
        return f"{path_parts[-2]}_{path_parts[-1]}"
    else:
        return parsed.path.strip('/').replace('/', '_')


def parse_spec_file(spec_path: Path) -> List[Tuple[str, Optional[str]]]:
    """
    Parse a spec file containing repository URLs and optional commit hashes.
    
    Format:
    - {repo_url}
    - {repo_url} {commit_hash}
    
    Args:
        spec_path: Path to the spec file
        
    Returns:
        List of tuples (repo_url, commit_hash), where commit_hash can be None
    """
    repos = []
    
    if not spec_path.exists():
        raise FileNotFoundError(f"Spec file not found: {spec_path}")
    
    with open(spec_path, 'r') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line or line.startswith('#'):
                continue
                
            parts = line.split()
            if len(parts) == 1:
                repos.append((parts[0], None))
            elif len(parts) == 2:
                repos.append((parts[0], parts[1]))
            else:
                print(f"Warning: Invalid format in line {line_num}: {line}")
                continue
                
    return repos





def clone_or_update_repo(
    repo_url: str, 
    target_commit: Optional[str], 
    repo_path: Path,
    current_lock_data: Dict[str, Any]
) -> Tuple[bool, Optional[str]]:
    """
    Clone or update a repository to the specified commit.
    
    Returns:
        Tuple of (success, actual_commit_hash)
    """
    repo_id = get_repo_id(repo_url)
    
    # Check if repository exists
    if repo_path.exists():
        current_hash = get_commit_hash(repo_path)
        
        # If we have a target commit and it's different from current, checkout the target
        if target_commit and current_hash and target_commit != current_hash:
            print(f"Repository {repo_id} exists but is at commit {current_hash}")
            print(f"Checking out target commit {target_commit}...")
            if checkout_commit(repo_path, target_commit):
                print(f"Successfully checked out commit {target_commit} for {repo_id}")
                return True, target_commit
            else:
                print(f"Failed to checkout target commit for {repo_id}, keeping current state")
                return False, current_hash
        else:
            print(f"Repository {repo_id} already exists at {repo_path}")
            return True, current_hash
    
    # Clone the repository
    print(f"Cloning {repo_url} into {repo_path}...")
    try:
        # Use unshallow clone if we need to checkout a specific commit
        clone_args = ["git", "clone"]
        if not target_commit:
            clone_args.extend(["--depth", "1"])
        clone_args.extend([repo_url, str(repo_path)])

        subprocess.run(clone_args, check=True, capture_output=True, text=True)
        print(f"Successfully cloned {repo_id}")

        # If we have a target commit, checkout that specific commit
        if target_commit:
            print(f"Checking out target commit {target_commit} for {repo_id}...")
            if checkout_commit(repo_path, target_commit):
                print(f"Successfully checked out commit {target_commit} for {repo_id}")
                return True, target_commit
            else:
                print(f"Failed to checkout target commit for {repo_id}, keeping HEAD")

        # Get current commit hash for newly cloned repo
        commit_hash = get_commit_hash(repo_path)
        if commit_hash:
            return True, commit_hash
        else:
            return False, None

    except subprocess.CalledProcessError as e:
        print(f"Error cloning {repo_id}: {e.stderr}")
        return False, None


def prepare_environment(
    spec_file_path: str,
    lock_file_path: Optional[str] = None,
    base_path: str = "codebases"
) -> List[Tuple[GitBenchRepo, Codebase]]:
    """
    Prepare an environment based on a spec file containing repository URLs.
    
    Args:
        spec_file_path: Path to the spec file containing repository URLs
        lock_file_path: Optional path to the lock file. If not provided, 
                       will use spec_file_path with .lock extension
        base_path: Base directory where repositories will be cloned
        
    Returns:
        List of tuples containing (GitBenchRepo, Codebase) for each repository
    """
    spec_path = Path(spec_file_path)
    
    if lock_file_path is None:
        lock_file_path = str(spec_path.with_suffix('.lock'))
    lock_path = Path(lock_file_path)
    
    base_dir = Path(base_path)
    base_dir.mkdir(exist_ok=True, parents=True)
    
    # Parse spec file
    repo_specs = parse_spec_file(spec_path)
    print(f"Found {len(repo_specs)} repositories in spec file")
    
    # Load existing lock file
    lock_data = LockFile.load(lock_path)
    
    # Prepare results
    results = []
    
    for repo_url, spec_commit in repo_specs:
        repo_id = get_repo_id(repo_url)
        repo_path = base_dir / repo_id
        
        # Determine target commit: spec file takes precedence over lock file
        target_commit = spec_commit
        if target_commit is None:
            existing_repo = lock_data.get_repo(repo_id)
            if existing_repo:
                target_commit = existing_repo.commit
        
        print(f"\n--- Processing repository: {repo_id} ---")
        print(f"URL: {repo_url}")
        if target_commit:
            print(f"Target commit: {target_commit}")
        
        # Clone or update repository
        existing_repo_data = lock_data.get_repo(repo_id)
        success, actual_commit = clone_or_update_repo(
            repo_url, target_commit, repo_path, existing_repo_data.model_dump() if existing_repo_data else {}
        )
        
        if not success:
            print(f"Failed to prepare repository {repo_id}, skipping...")
            continue
            
        if not actual_commit:
            print(f"Could not determine commit hash for {repo_id}, skipping...")
            continue
        
        # Create Codebase
        try:
            print(f"Creating codebase for {repo_id}...")
            codebase = Codebase.new(
                id=repo_id,
                dir=repo_path,
                url=repo_url,
                lazy=True  # Use lazy loading for better performance
            )
            
            # Initialize chunks for stats calculation
            codebase.init_chunks()
            
            # Generate stats
            print(f"Generating stats for {repo_id}...")
            stats = CodebaseStats.from_codebase(codebase)
            
            # Create GitBenchRepo
            git_bench_repo = GitBenchRepo(
                url=repo_url,
                commit=actual_commit,
                stats=StatsDigest.from_codebase_stats(stats)
            )
            
            results.append((git_bench_repo, codebase))
            
            # Update lock data
            lock_data.set_repo(repo_id, git_bench_repo)
            
            print(f"Successfully prepared {repo_id}")
            print(f"  Files: {stats.num_files}")
            print(f"  Lines: {stats.num_lines}")
            print(f"  Primary language: {stats.primary_language}")
            
        except Exception as e:
            print(f"Error creating codebase for {repo_id}: {e}")
            continue
    
    # Save updated lock file
    lock_data.save(lock_path)
    
    print(f"\n=== Environment preparation complete ===")
    print(f"Successfully prepared {len(results)} repositories")
    
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Prepare benchmark environments from repository specification files"
    )
    parser.add_argument(
        "spec_file", 
        help="Path to the spec file containing repository URLs"
    )
    parser.add_argument(
        "--lock-file", 
        help="Path to the lock file (default: spec_file with .lock extension)"
    )
    parser.add_argument(
        "--base-path", 
        default="codebases",
        help="Base directory for cloning repositories (default: codebases)"
    )
    parser.add_argument(
        "--stats-only",
        action="store_true",
        help="Only show statistics, don't process repositories"
    )
    
    args = parser.parse_args()
    
    if args.stats_only:
        # Just show what would be processed
        spec_path = Path(args.spec_file)
        if not spec_path.exists():
            print(f"Error: Spec file not found: {args.spec_file}")
            exit(1)
            
        repos = parse_spec_file(spec_path)
        print(f"Found {len(repos)} repositories in {args.spec_file}:")
        for i, (url, commit) in enumerate(repos, 1):
            print(f"  {i}. {url}" + (f" @ {commit}" if commit else ""))
        
        lock_path = Path(args.lock_file) if args.lock_file else spec_path.with_suffix('.lock')
        if lock_path.exists():
            lock_data = LockFile.load(lock_path)
            print(f"\nLock file {lock_path} contains {len(lock_data.repositories)} entries")
    else:
        # Process repositories
        try:
            results = prepare_environment(
                spec_file_path=args.spec_file,
                lock_file_path=args.lock_file,
                base_path=args.base_path
            )
            
            print(f"\n=== Final Summary ===")
            total_files = sum(repo.stats.num_files for repo, _ in results)
            total_lines = sum(repo.stats.num_lines for repo, _ in results) 
            total_tokens = sum(repo.stats.num_tokens for repo, _ in results)
            
            print(f"Successfully prepared {len(results)} repositories")
            print(f"Total files: {total_files:,}")
            print(f"Total lines: {total_lines:,}")
            print(f"Total tokens: {total_tokens:,}")
            
        except Exception as e:
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()
            exit(1)

