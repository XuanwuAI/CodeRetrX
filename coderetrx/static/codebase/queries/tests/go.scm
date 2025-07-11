; Functions names start with `Test`
(
  (
    (function_declaration name: (_) @run
      (#match? @run "^Test.*"))
  ) @_
  (#set! tag go-test)
)

; `go:generate` comments
(
    ((comment) @_comment @run
    (#match? @_comment "^//go:generate"))
    (#set! tag go-generate)
)

; `t.Run`
(
  (
    (call_expression
      function: (
        selector_expression
        field: _ @run @_name
        (#eq? @_name "Run")
      )
      arguments: (
        argument_list
        .
        (interpreted_string_literal) @_subtest_name
        .
        (func_literal
          parameters: (
            parameter_list
            (parameter_declaration
              name: (identifier) @_param_name
              type: (pointer_type
                (qualified_type
                  package: (package_identifier) @_pkg
                  name: (type_identifier) @_type
                  (#eq? @_pkg "testing")
                  (#eq? @_type "T")
                )
              )
            )
          )
        ) @_second_argument
      )
    )
  ) @_
  (#set! tag go-subtest)
)

; Functions names start with `Benchmark`
(
  (
    (function_declaration name: (_) @run @_name
      (#match? @_name "^Benchmark.+"))
  ) @_
  (#set! tag go-benchmark)
)

; Functions names start with `Fuzz`
(
  (
    (function_declaration name: (_) @run @_name
      (#match? @_name "^Fuzz"))
  ) @_
  (#set! tag go-fuzz)
)

