; Macros `describe`, `test` and `property`.
; This matches the ExUnit test style.
(
    (call
        target: (identifier) @run (#any-of? @run "describe" "test" "property")
    ) @_elixir-test
    (#set! tag elixir-test)
)

; Modules containing at least one `describe`, `test` and `property`.
; This matches the ExUnit test style.
(
    (call
        target: (identifier) @run (#eq? @run "defmodule")
        (do_block
            (call target: (identifier) @_keyword (#any-of? @_keyword "describe" "test" "property"))
        )
    ) @_elixir-module-test
    (#set! tag elixir-module-test)
)