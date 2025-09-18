; tree-sitter query to extract Elixir dependencies

; Match alias statements
(call
  target: (identifier) @directive (#eq? @directive "alias")
  (arguments (alias) @module)
) @alias_stmt

; Match import statements
(call
  target: (identifier) @directive (#eq? @directive "import")
  (arguments (alias) @module)
) @import_stmt

; Match require statements
(call
  target: (identifier) @directive (#eq? @directive "require")
  (arguments (alias) @module)
) @require_stmt