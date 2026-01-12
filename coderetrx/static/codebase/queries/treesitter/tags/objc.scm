; Objective-C language tags query

; Class interface declarations (@interface)
(class_interface
  name: (identifier) @name.definition.class) @definition.class

; Class implementation (@implementation)
(class_implementation
  name: (identifier) @name.definition.class) @definition.class

; Protocol declarations (@protocol)
(protocol_declaration
  name: (identifier) @name.definition.interface) @definition.interface

; Method declarations in @interface
(class_interface
  (method_declaration
    (identifier) @name.definition.method)) @definition.method

; Method definitions in @implementation
(class_implementation
  (implementation_definition
    (method_definition
      (identifier) @name.definition.method))) @definition.method

; Property declarations (@property)
(property_declaration
  (struct_declaration
    (struct_declarator
      (pointer_declarator
        (identifier) @name.definition.variable)))) @definition.variable

(property_declaration
  (struct_declaration
    (struct_declarator
      (identifier) @name.definition.variable))) @definition.variable

; Function definitions (C-style functions)
(function_definition
  declarator: (function_declarator
    declarator: (identifier) @name.definition.function)) @definition.function

(function_definition
  declarator: (pointer_declarator
    declarator: (function_declarator
      declarator: (identifier) @name.definition.function))) @definition.function

; Typedef declarations
(type_definition
  declarator: (type_identifier) @name.definition.type) @definition.type

; Enum declarations
(enum_specifier
  name: (type_identifier) @name.definition.type) @definition.type

; Struct declarations
(struct_specifier
  name: (type_identifier) @name.definition.class
  body: (_)) @definition.class

; Import and include statements (#import and #include)
(preproc_include) @import

; Method calls (message expressions)
(call_expression
  function: (identifier) @name.reference.call) @reference.call

; Variable definitions - local variables
(declaration
  declarator: (identifier) @name.definition.variable) @definition.variable

; Variable definitions - one level of nesting
(declaration
  declarator: (_
    declarator: (identifier) @name.definition.variable)) @definition.variable

; Variable definitions - two levels of nesting
(declaration
  declarator: (_
    declarator: (_
      declarator: (identifier) @name.definition.variable))) @definition.variable
