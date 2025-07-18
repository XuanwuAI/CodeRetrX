; Definitions

; * modules and protocols
(call
  target: (identifier) @ignore
  (arguments (alias) @name.definition.module)
  (#match? @ignore "^(defmodule|defprotocol)$")) @definition.module

; * functions/macros
(call
  target: (identifier) @ignore
  (arguments
    [
      ; zero-arity functions with no parentheses
      (identifier) @name.definition.function
      ; regular function clause
      (call target: (identifier) @name.definition.function)
      ; function clause with a guard clause
      (binary_operator
        left: (call target: (identifier) @name.definition.function)
        operator: "when")
    ])
  (#match? @ignore "^(def|defp|defdelegate|defguard|defguardp|defmacro|defmacrop|defn|defnp)$")) @definition.function

; References

; ignore calls to kernel/special-forms keywords
(call
  target: (identifier) @ignore
  (#match? @ignore "^(def|defp|defdelegate|defguard|defguardp|defmacro|defmacrop|defn|defnp|defmodule|defprotocol|defimpl|defstruct|defexception|defoverridable|alias|case|cond|else|for|if|import|quote|raise|receive|require|reraise|super|throw|try|unless|unquote|unquote_splicing|use|with)$"))

; ignore module attributes
(unary_operator
  operator: "@"
  operand: (call
    target: (identifier) @ignore))

; * function call
(call
  target: [
   ; local
   (identifier) @name.reference.call
   ; remote
   (dot
     right: (identifier) @name.reference.call)
  ]) @reference.call

; * pipe into function call
(binary_operator
  operator: "|>"
  right: (identifier) @name.reference.call) @reference.call

; * modules
(alias) @name.reference.module @reference.module

(call
  target: (identifier)@_cap
  (#any-of? @_cap "import" "use" "require"))@import