/**
 * @name Python Functions with last statement
 * @description Find all functions and their last statement
 * @kind table
 * @id codeql/python-functions-laststmt
 */

import python

from Function func, Stmt stmt
where
  stmt = func.getBody().getLastItem()
select 
  func.getLocation().getFile().getRelativePath(), 
  func.getQualifiedName(), 
  func.getLocation().getStartLine(), 
  stmt.getLocation().getEndLine(), 
  func.getLocation().getStartColumn(),
  func.getLocation().getEndColumn()