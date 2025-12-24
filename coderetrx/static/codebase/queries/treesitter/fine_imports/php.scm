;; PHP import statements
;; Basic namespace imports (use statements)
(namespace_use_declaration
  (namespace_use_clause
    (qualified_name) @module))

;; require/include statements with string literals
(require_expression
  (string) @module)

(require_once_expression
  (string) @module)

(include_expression
  (string) @module)

(include_once_expression
  (string) @module)
