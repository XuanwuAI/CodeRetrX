(struct_specifier name: (type_identifier) @name.definition.class body:(_)) @definition.class

(declaration type: (union_specifier name: (type_identifier) @name.definition.class)) @definition.class

(function_definition
  declarator: (function_declarator
    declarator: (identifier) @name.definition.function)) @definition.function

(function_definition
  declarator: (pointer_declarator
    declarator: (function_declarator
      declarator: (identifier) @name.definition.function))) @definition.function

(type_definition declarator: (type_identifier) @name.definition.type) @definition.type

(enum_specifier name: (type_identifier) @name.definition.type) @definition.type

(preproc_include) @import

; Variable definitions - all declarations
(declaration
  declarator: (identifier) @name.definition.variable) @definition.variable

(declaration
  declarator: (init_declarator
    declarator: (identifier) @name.definition.variable)) @definition.variable

(declaration
  declarator: (pointer_declarator
    declarator: (identifier) @name.definition.variable)) @definition.variable
