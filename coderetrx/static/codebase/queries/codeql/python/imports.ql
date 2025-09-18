/**
 * @name Python Imports
 * @description Find all import statements in Python code
 * @kind table
 * @id codeql/python-imports
 */

import python

from Import imp
where exists(imp.getLocation()) and exists(imp.getAnImportedModuleName())
select imp.getLocation().getFile().getRelativePath(), imp.getAnImportedModuleName(),
       imp.getLocation().getStartLine(), imp.getLocation().getEndLine(),
       imp.getLocation().getStartColumn(), imp.getLocation().getEndColumn()