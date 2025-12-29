; C++ language tags query - simplified version

(struct_specifier name: (type_identifier) @name.definition.class body:(_)) @definition.class

(declaration type: (union_specifier name: (type_identifier) @name.definition.class)) @definition.class

(function_definition
  declarator: (function_declarator
    declarator: (identifier) @name.definition.function)) @definition.function

(function_definition
  declarator: (pointer_declarator
    declarator: (function_declarator
      declarator: (identifier) @name.definition.function))) @definition.function

(function_definition
  declarator: (function_declarator
    declarator: (field_identifier) @name.definition.function)) @definition.function

(function_definition
  declarator: (function_declarator
    declarator: (qualified_identifier scope: (namespace_identifier) @scope name: (identifier) @name.definition.method))) @definition.method

(function_definition
  declarator: (reference_declarator
    (function_declarator
      declarator: (qualified_identifier scope: (namespace_identifier) @scope name: (identifier) @name.definition.method)))) @definition.method

(type_definition declarator: (type_identifier) @name.definition.type) @definition.type

(enum_specifier name: (type_identifier) @name.definition.type) @definition.type

(class_specifier name: (type_identifier) @name.definition.class) @definition.class

(preproc_include) @import

; Variable definitions - hybrid approach
; Combines universal wildcard matching with specific function pointer patterns

; Level 0: Direct identifier
(declaration
  declarator: (identifier) @name.definition.variable) @definition.variable

; Level 1: One level of nesting (covers most cases)
(declaration
  declarator: (_
    declarator: (identifier) @name.definition.variable)) @definition.variable

; Level 2: Two levels of nesting
(declaration
  declarator: (_
    declarator: (_
      declarator: (identifier) @name.definition.variable))) @definition.variable

; Level 3: Three levels of nesting
(declaration
  declarator: (_
    declarator: (_
      declarator: (_
        declarator: (identifier) @name.definition.variable)))) @definition.variable

; Level 4: Four levels of nesting
(declaration
  declarator: (_
    declarator: (_
      declarator: (_
        declarator: (_
          declarator: (identifier) @name.definition.variable))))) @definition.variable

; Special case: Function pointers with parenthesized_declarator
; These need explicit handling because function_declarator is at the top level
(declaration
  declarator: (function_declarator
    declarator: (parenthesized_declarator
      (_
        declarator: (identifier) @name.definition.variable)))) @definition.variable

(declaration
  declarator: (function_declarator
    declarator: (parenthesized_declarator
      (_
        declarator: (_
          declarator: (identifier) @name.definition.variable))))) @definition.variable
