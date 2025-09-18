;; Static imports
(import_statement source: (string) @module)

;; Require
(call_expression
    function: (identifier) @func
    arguments: (arguments (string) @module)
    (#eq? @func "require"))

;; Re-exports
(export_statement source: (string) @module)