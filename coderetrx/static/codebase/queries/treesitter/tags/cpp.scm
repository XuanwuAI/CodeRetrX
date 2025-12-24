(struct_specifier name: (type_identifier) @name.definition.class body:(_)) @definition.class

(declaration type: (union_specifier name: (type_identifier) @name.definition.class)) @definition.class

(function_declarator declarator: (identifier) @name.definition.function) @definition.function

(function_declarator declarator: (field_identifier) @name.definition.function) @definition.function

(function_declarator declarator: (qualified_identifier scope: (namespace_identifier) @scope name: (identifier) @name.definition.method)) @definition.method

(type_definition declarator: (type_identifier) @name.definition.type) @definition.type

(enum_specifier name: (type_identifier) @name.definition.type) @definition.type

(class_specifier name: (type_identifier) @name.definition.class) @definition.class

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

(declaration
  declarator: (reference_declarator
    (identifier) @name.definition.variable)) @definition.variable
