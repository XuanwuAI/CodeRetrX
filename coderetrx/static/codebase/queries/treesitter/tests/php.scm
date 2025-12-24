;; PHPUnit test methods (methods starting with test)
;; Only match test methods, not test classes, to allow non-test methods in test classes
(
    (method_declaration
        name: (name) @run @_test_method_name
        (#match? @_test_method_name "^test")
    ) @_php-test-method
    (#set! tag php-test-method)
)
