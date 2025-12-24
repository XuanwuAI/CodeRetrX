;; Class definitions
(class_declaration
  name: (name) @name.definition.class) @definition.class

;; Interface definitions
(interface_declaration
  name: (name) @name.definition.interface) @definition.interface

;; Trait definitions
(trait_declaration
  name: (name) @name.definition.class) @definition.class

;; Function definitions
(function_definition
  name: (name) @name.definition.function) @definition.function

;; Method definitions
(method_declaration
  name: (name) @name.definition.method) @definition.method

;; Namespace imports
(namespace_use_declaration
  (namespace_use_clause
    (qualified_name) @name.import)) @import

;; Object creation (class references)
(object_creation_expression
  [
    (qualified_name (name) @name.reference.class)
    (variable_name (name) @name.reference.class)
  ]) @reference.class

;; Function calls
(function_call_expression
  function: [
    (qualified_name (name) @name.reference.call)
    (variable_name (name)) @name.reference.call
  ]) @reference.call

;; Static method calls
(scoped_call_expression
  name: (name) @name.reference.call) @reference.call

;; Member method calls
(member_call_expression
  name: (name) @name.reference.call) @reference.call

;; Property declarations (class variables)
(property_declaration
  (property_element
    (variable_name) @name.definition.variable)) @definition.variable
