/* Copyright (C) 2008-2014 University of Massachusetts Amherst.
   This file is part of "FACTORIE" (Factor graphs, Imperative, Extensible)
   http://factorie.cs.umass.edu, http://github.com/factorie
   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0
   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License. */
package cc.factorie.app.nlp.hcoref

/**
 * @author John Sullivan
 */
trait PairGenerator[Vars <: NodeVariables[Vars]] {
  def nextContext:(Node[Vars], Node[Vars])
  def iterations:Int
  def mentions:Iterable[Node[Vars]]

  def contexts:Iterable[(Node[Vars], Node[Vars])] = new Iterator[(Node[Vars], Node[Vars])] {

    var index = 0

    def hasNext: Boolean = index < iterations

    def next(): (Node[Vars], Node[Vars]) = if(hasNext) {
      index += 1
      nextContext
    } else {
      throw new NoSuchElementException("Max iterations exceeded %d" format iterations)
    }
  }.toStream
}