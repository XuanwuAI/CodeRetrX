import asyncio
import os
from typing import List
from pathlib import Path
from pydantic import BaseModel, Field
from coderetrx.static.ripgrep import ripgrep_glob  # type: ignore
from coderetrx.tools.base import BaseTool
from coderetrx.utils.path import safe_join


class ListDirResult(BaseModel):
    path: str = Field(description="Path of the directory")
    is_dir: bool = Field(description="Whether the path is a directory")
    size: int = Field(description="Size of the file in bytes")
    parent: str = Field(description="Parent directory path")

    @classmethod
    def repr(cls, entries: List["ListDirResult"]) -> str:
        """
        Serialize a directory structure to a tree-like string output
        """

        def _list_to_tree_string(
            entries: list["ListDirResult"],
            root_path: str = "",
            indent="",
            is_root=False,
        ):
            """
            Serialize a list of directory structures to a tree-like string output

            Arguments:
            entries -- A list of directory structures
            root_path -- The root directory path
            indent -- The current indentation
            is_root -- Whether the current item is the root directory
            """
            if not entries:
                return []
            tree_lines = []

            # get direct children of the current root path
            direct_children = [entry for entry in entries if entry.parent == root_path]

            # sort the direct children by name
            direct_children.sort(key=lambda x: (not x.is_dir, x.path))

            for i, entry in enumerate(direct_children):
                is_last_item = i == len(direct_children) - 1

                # determine the connector symbol
                if is_root:
                    connector = ""
                else:
                    connector = "└── " if is_last_item else "├── "

                # determine the next indentation
                if is_root:
                    next_indent = ""
                else:
                    next_indent = indent + ("    " if is_last_item else "│   ")

                name = entry.path.split("/")[-1]

                # add the current item to the tree lines
                cur_item = f"{indent}{connector}{name}"
                if entry.is_dir:
                    cur_item += "/"
                tree_lines.append(cur_item)

                # if the current item is a directory, recursively call the function to add its children
                if entry.is_dir:
                    sub_tree_lines = _list_to_tree_string(
                        entries,
                        entry.path,
                        next_indent,
                    )
                    tree_lines.extend(sub_tree_lines)

            return tree_lines

        min_entry = min(entries, key=lambda x: len(x.path.split("/")))
        min_entreis = [
            entry
            for entry in entries
            if len(entry.path.split("/")) == len(min_entry.path.split("/"))
        ]
        root_path = min_entry.path if len(min_entreis) == 1 else min_entry.parent

        tree_lines = _list_to_tree_string(entries, root_path, is_root=True)
        result = root_path + "\n" + "\n".join(tree_lines)
        return result


class ListDirTool(BaseTool):
    name = "list_dir"
    description = (
        "Retrieves detailed directory structure information\n"
        "Output includes:\n"
        "1. Relative path of each item\n"
        "2. Type indicator (file/directory)\n"
        "3. File size in bytes\n"
        "4. Child count (directories only)\n\n"
        "Requirements:\n"
        "- Requires absolute path\n"
        "- Directory must exist and be readable\n"
    )
    inputs = {
        "directory_path": {
            "description": "path of directory to list\nExample: 'src/main/resources'",
            "type": "string",
        },
        "limit": {
            "description": "Maximum number of items to return",
            "type": "integer",
        },
    }
    output_type = "string"

    def forward(self, directory_path: str, limit: int) -> str:
        """Synchronous wrapper for async _run method."""
        return self.run_sync(directory_path=directory_path, limit=limit)

    async def _run(self, directory_path: str = "/", limit: int = 100) -> list[ListDirResult]:
        results = []
        directory_path = directory_path.lstrip("/")
        full_directory_path: Path = safe_join(self.repo_path, directory_path)

        if not full_directory_path.exists():
            return [ListDirResult(path="Path Not Exists", is_dir=True, size=0, parent="")]

        file_entries = await ripgrep_glob(
            full_directory_path, "*", extra_argv=["-g", "!.git"]
        )
        full_entry_paths: list[Path] = []
        parent_added = set([full_directory_path])
        for entry in file_entries:
            full_entry_path = full_directory_path / Path(entry)
            cur_parent = full_entry_path.parent
            while cur_parent.is_relative_to(full_directory_path):
                if cur_parent in parent_added:
                    break
                parent_added.add(cur_parent)
                full_entry_paths.append(cur_parent)
                cur_parent = cur_parent.parent
            full_entry_paths.append(full_entry_path)

        # sort by layer, file after dir, alpha
        full_entry_paths.sort(
            key=lambda p: (len(p.parts), not p.is_dir(), str(p).lower())
        )
        # Handle empty directory case
        if not full_entry_paths:
            return results 
        # calculate the number of entries in the first layer
        min_len = len(
            full_directory_path.parts
        )  # Start with directory path's parts length
        if full_entry_paths:  # This check is redundant now but kept for safety
            min_len = min(len(p.parts) for p in full_entry_paths)
        first_layer_entry_num = len(
            [p for p in full_entry_paths if len(p.parts) == min_len]
        )
        # ensure that the first layer is always included, unless threshold is achieved:
        # defualt display number is limit; if first_layer_entry_num > limit, then show them as many as possible, unless unless threshold is achieved
        threshold_num = 300
        display_count = min(max(limit, first_layer_entry_num), threshold_num)
        # for the rest of the entries, diplay #display_omit_count of them
        display_omit_count = max(50, int(display_count * 0.5))
        count = 0
        for full_entry_path in full_entry_paths:
            entry_path = full_entry_path.relative_to(self.repo_path)
            is_dir = full_entry_path.is_dir()
            parent = entry_path.parent
            result = ListDirResult(
                path=str(entry_path),
                is_dir=is_dir,
                size=full_entry_path.stat().st_size if not is_dir else 0,
                parent=str(parent),
            )

            if count < display_count:
                results.append(result)
                count += 1
            elif count < display_count + display_omit_count:
                fake_result = ListDirResult(
                    path=result.parent + "/(Omitted...)",
                    is_dir=result.is_dir,
                    size=0,  # set to default 0, convenient to check duplicates
                    parent=result.parent,
                )
                # ensure that the fake result is not duplicated
                if fake_result not in results:
                    results.append(fake_result)
                    count += 1
            else:
                break

        return results
