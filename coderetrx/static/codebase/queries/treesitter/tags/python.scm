(class_definition
  name: (identifier) @name.definition.class) @definition.class

(function_definition
  name: (identifier) @name.definition.function) @definition.function

(call
  function: [
      (identifier) @name.reference.call
      (attribute
        attribute: (identifier) @name.reference.call)
  ]) @reference.call

; Handle imports
(import_statement) @import
(import_from_statement) @import

; Variable definitions - all assignment statements
(assignment
  left: (identifier) @name.definition.variable) @definition.variable

(assignment
  left: (pattern_list
    (identifier) @name.definition.variable)) @definition.variable

(assignment
  left: (attribute) @name.definition.variable) @definition.variable
