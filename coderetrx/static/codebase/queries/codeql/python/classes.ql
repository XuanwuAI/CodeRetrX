/**
 * @name Python Classes with last statement
 * @description Find all class definitions and their last statement
 * @kind table
 * @id codeql/python-classes-laststmt
 */

import python

from Class cls, Stmt stmt
where
  stmt = cls.getBody().getLastItem()
select 
  cls.getLocation().getFile().getRelativePath(), 
  cls.getQualifiedName(), 
  cls.getLocation().getStartLine(), 
  stmt.getLocation().getEndLine(), 
  cls.getLocation().getStartColumn(), 
  cls.getLocation().getEndColumn()