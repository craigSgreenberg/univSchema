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
package cc.factorie.app.nlp.embeddings
import java.nio.charset.Charset


object WordVec {
  def main(args: Array[String]) {
    val opts = new EmbeddingOpts
    opts.parse(args)
    println("Default Charset of this JVM=" + Charset.defaultCharset());
    println("User Provided Charset for this project=" + opts.encoding.value)
    val wordEmbedding = opts.options.value match {
      case 1 => new HierarchicalSoftmax(opts)
      case 2 => new NegativeSampling(opts)
      case 3 => new NeighborhoodClassifier(opts)
    }
    println(wordEmbedding.D, wordEmbedding.adaGradRate, wordEmbedding.opts.regularizer.value, wordEmbedding.opts.epochs.value, wordEmbedding.opts.negative.value, wordEmbedding.opts.threads.value)
    println(opts.writeOutput.value)
    println(opts.treeFile.value)
    val st1 = System.currentTimeMillis()
    wordEmbedding.buildVocab()
    val st = System.currentTimeMillis()
    println("time taken to create vocab : " + (st - st1) / 1000.0)
    wordEmbedding.learnEmbeddings()
    val en = System.currentTimeMillis() - st
    println("time taken to learn embeddings : " + en / 1000.0)
    //val st2 = System.currentTimeMillis()
    //write test output
    //wordEmbedding.store()
    //val en1 = System.currentTimeMillis() - st2
    //println("time taken to store embeddings :" + en1 / 1000.0)

  }
}
