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

import java.util

import cc.factorie.model.{ Parameters, Weights }
import cc.factorie.optimize.{HogwildTrainer, Trainer, AdaGradRDA}
import cc.factorie.la.{SparseHashTensor1, SparseBinaryTensor1, DenseTensor1}
import cc.factorie.util.Threading
import java.io._
import java.util.zip.{ GZIPOutputStream, GZIPInputStream }
import scala.collection.mutable
import scala.util.control.Breaks._
import scala.util.Random
import java.util.HashMap
import scala.collection.mutable.ArrayBuffer
import java.util.HashSet

abstract class UniversalSchemaModel(val opts: EmbeddingOpts) extends Parameters {
  //val entityPairFeatures = new mutable.HashMap[Int, ArrayBuffer[Int]]()
  val entityPairFeatures = new mutable.HashMap[Int, SparseBinaryTensor1]()
  var classifierWeights: Seq[Weights] = null
  val testRels: util.HashSet[String] = new util.HashSet[String]()
  var processed = 0
  // Algo related
  val rand = new Random(0)
  val D = opts.dimension.value // default value is 200

  var entityCount:Int = -1
  var entPairSize: Int = -1 // no. of entity pairs
  var relationSize: Int = -1 // no. of relations
  var trainingExamplesSize : Int = -1 //no. of training examples

  protected val entPairKey = new util.HashMap[String, Int]()
  protected val entityVocab = new util.HashMap[String, Int]()
  protected val relationKey = new util.HashMap[String, Int]()
  protected val reverseRelationKey = new util.HashMap[Int, String]()
  // entPair, e1, e2, relation
  protected var trainingExamples = List[(Int, Int, Int, Int)]()

  var tree = new mutable.HashMap[Int, treePosition]()
  protected val threads = opts.threads.value //  default value is 12
  protected val adaGradDelta = opts.delta.value // default value is 0.1
  val adaGradRate = opts.rate.value //  default value is 0.025
  val hierarchicalSoftMax = opts.hierSoftMax.value

  // IO Related
  val corpus = opts.corpus.value // corpus input filename. Code takes cares of .gz extension 
  protected val outputFilename = opts.output.value // embeddings output filename
  protected val encoding = opts.encoding.value // Default is UTF8
  // data structures
  //protected var vocab: VocabBuilder = null
  protected var trainer: LiteHogwildTrainer = null // modified version of factorie's hogwild trainer for speed by removing logging and other unimportant things. Expose processExample() instead of processExamples()
  protected var optimizer: AdaGradRDA = null

  var weights: Seq[Weights] = null // EMBEDDINGS . Will be initialized in learnEmbeddings() after buildVocab() is called first
  var nodeWeights: Seq[Weights] = null



  def buildBinaryTree(){
    val treeFile =   io.Source.fromInputStream(new FileInputStream(opts.treeFile.value), encoding).getLines()
    while (treeFile.hasNext) {
     val Array(rel, nl, code) =  treeFile.next().split("\t")
      tree.getOrElseUpdate(relationKey.get(rel), new treePosition(rel, nl.split(" ").toSeq.map(c => c.toInt).toArray, code.split(" ").toSeq.map(c => c.toInt).toArray))
    }
  }

  def buildVocab(): Unit ={
    val corpusLineItr = corpus.endsWith(".gz") match {
      case true => io.Source.fromInputStream(new GZIPInputStream(new FileInputStream(corpus)), encoding).getLines()
      case false => io.Source.fromInputStream(new FileInputStream(corpus), encoding).getLines()
    }
    while (corpusLineItr.hasNext) {
      val line = corpusLineItr.next()
      val (ep, e1, e2, rel, label) = if(opts.parseTsv.value) parseTsv(line) else parseArvind(line)
      if(!entPairKey.containsKey(ep))  entPairKey.put(ep, entPairKey.size())
      if(!entityVocab.containsKey(e1))  entityVocab.put(e1, entityVocab.size())
      if(!entityVocab.containsKey(e2))  entityVocab.put(e2, entityVocab.size())
      if(!relationKey.containsKey(rel)) relationKey.put(rel, relationKey.size())
      trainingExamples = (entPairKey.get(ep), entityVocab.get(e1), entityVocab.get(e2), relationKey.get(rel)) :: trainingExamples
      reverseRelationKey.put(relationKey.get(rel), rel)
    }

    entPairSize = entPairKey.size
    entityCount = entityVocab.size
    relationSize = relationKey.size
    trainingExamplesSize = trainingExamples.size
    /*
    for(i <- 0 until relationSize){
      //println(i, positives(i).size)
      for(j <- 0 until  entPairSize){
        if(!(positives(i).contains(j))) negatives(i) = negatives.getOrElseUpdate(i, ArrayBuffer[Int]()) += j
      }
      //println(i, negatives(i).size)
    }
    */
    println("Number of entities: ", entityCount)
    println("Number of entity pairs: ", entPairSize)
    println("Number of relations: ", relationSize, reverseRelationKey.size)
    println("Number of training examples: ", trainingExamplesSize)
    if(opts.options.value == 1)  buildBinaryTree()
  }

  /**
   * assumes arvind format : [e1,e2\trelation\tscore]
   * @param line a line from inputFile.arvind
   * @return entPair, e1, e2, rel, label
   */
  def parseArvind(line: String): (String, String, String, String, String) = {
    val Array(entities, relation, score) = line.split("\t")
    val Array(e1, e2) = entities.split(",")
    (entities, e1, e2, relation, score)
  }

  /**
   * assumes a tsv formatted as [ e1 rel e2 ], score is inferred as 1.0
   * @param line a line from inputFile.tsv
   * @return entPair, e1, e2, rel, label=1.0
   */
  def parseTsv(line: String): (String, String, String, String, String) = {
    val Array(e1, rel, e2) = line.split("\t")
    (s"$e1,$e2", e1, e2, rel, "1.0")
  }

  def evaluate(file: String, iter: Int): Double = {
    var notfound = 0
    val  corpusLineItr = io.Source.fromInputStream(new FileInputStream(file), encoding).getLines
    val ans = scala.collection.mutable.Map[Int, ArrayBuffer[(Double, Boolean)]]()
    //val p = if(opts.writeOutput                                                                                                                                        .value)  new PrintWriter(new File(file + "_" + "output" + "_" + D.toString + "_" + adaGradRate.toString + "_" + opts.regularizer.value.toString + "_" + opts.negative.value.toString + "_" + iter.toString + "_"  + opts.hinge.value.toString + "_" + opts.wsabie.value.toString + "_" + opts.margin.value.toString))
    val fileName = file + "_" + "output" +  "_" + D.toString + "_" + adaGradRate.toString + "_" + opts.regularizer.value.toString + "_" + opts.negative.value.toString + "_" + iter.toString + "_"  + opts.hinge.value.toString + "_" + opts.wsabie.value.toString + "_" + opts.margin.value.toString + "_" + opts.treeFile.value.toString.split("/").reverse(0)
    println(fileName)
    val p = if(opts.writeOutput.value)  new PrintWriter(new File(fileName))
    try{
      while (corpusLineItr.hasNext) {
        val line = corpusLineItr.next()
        val (ep, e1, e2, rel, label) = if(opts.parseTsv.value) parseTsv(line) else parseArvind(line)
        var truth = true
        if(label == "0")  truth = false
        if(entPairKey.containsKey(ep)) {
          val s = getScore(ep, rel)
          if(opts.writeOutput.value)  p.asInstanceOf[PrintWriter].write(rel + "\t0\t" + ep + "\t0\t" + s.toString + "\tmycode\n")
          //if(truth) k.write(rel + " 0 " + ep + " 1\n")

          ans(relationKey.get(rel)) = ans.getOrElseUpdate(relationKey.get(rel), ArrayBuffer[(Double, Boolean)]()) += ((s,truth))
        }
        else notfound += 1
      }
    }
    finally {
      if(opts.writeOutput.value)  p.asInstanceOf[PrintWriter].close()
    }
    var predictionSize = 0
    for(i <- ans.keys) {
      ans(i) = rand.shuffle(ans(i))
      predictionSize += ans(i).size
    }
    println("prediction size : ", predictionSize)
    println("not found: ", notfound)
    Evaluator.meanAveragePrecision(ans)
  }

  def getScore(ep : String, rel : String): Double ={
    getScore(entPairKey.get(ep), relationKey.get(rel))
  }

  // Component-2
  def learnEmbeddings(): Unit = {
    println("Learning Embeddings")
    optimizer = new AdaGradRDA(delta = adaGradDelta, rate = adaGradRate, l2 = opts.regularizer.value)
    weights = (0 until entPairSize).map(i => Weights(TensorUtils.setToRandom1(new DenseTensor1(D, 0), rand))) // initialized using wordvec random
    // for hierarchical softmax
    nodeWeights =  (0 until relationSize).map(i => Weights(TensorUtils.setToRandom1(new DenseTensor1(D, 0), rand)))
    // set for paragraph vector
    //parWeights =   (0 until docNum).map(i => Weights(TensorUtils.setToRandom1(new DenseTensor1(D, 0))))
    optimizer.initializeWeights(this.parameters)
    trainer = new LiteHogwildTrainer(weightsSet = this.parameters, optimizer = optimizer, nThreads = threads, maxIterations = Int.MaxValue)
    println("number of threads: ", threads)
    val threadIds = (0 until threads).map(i => i)
        //var word_count: Long = 0
    val groupSize = trainingExamplesSize % threads == 0 match {
          case true => trainingExamplesSize/threads
          case false =>  trainingExamplesSize/threads + 1
        }
    for(i <- 1 until (opts.epochs.value + 1)){ // number of iterations
          println("Training Iteration " , i , processed)
          processed = 0
          val st1 = System.currentTimeMillis()
          val it = rand.shuffle(trainingExamples).grouped(groupSize)
          //println("match ", threads, groupSize)
          var threadExamples = new ArrayBuffer[List[(Int, Int, Int, Int)]]()
          for(n <- 0 until threads)  threadExamples = threadExamples += it.next()
          val st = System.currentTimeMillis()
          println("comuting gradients " + (st - st1) / 1000.0)
          Threading.parForeach(threadIds, threads)(threadId => workerThread(threadExamples(threadId)))
          val st2 = System.currentTimeMillis()
          println("finished comuting gradients " + (st2 - st) / 1000.0)
          if(i % opts.evalautionFrequency.value == 0) {
            println("Dev MAP after " + i + " iterations: " + evaluate(opts.devFile.value, i))
            println("Test MAP after " + i + " iterations: " + evaluate(opts.testFile.value, i))
          }
    }
    println("Done learning embeddings. ")
    val out = if(opts.writeVecs.value)  new PrintWriter(new File("relationVecs"))
    try{
        if(opts.writeVecs.value){
          for(i <- 0 until relationSize)  out.asInstanceOf[PrintWriter].write(reverseRelationKey.get(i) + "\t" + nodeWeights(i).value.toString + "\n")
        }
    }
    finally {
      if(opts.writeVecs.value) out.asInstanceOf[PrintWriter].close()
    }
  }

  /**
   *
   * @param examples  entPair, e1, e2, relation
   */
  protected def workerThread(examples: List[(Int, Int, Int, Int)]): Unit = {
    //println("processing: " + examples.size)
    examples.foreach(example => process(example._1, example._4))
  }

  // override this function in your Embedding Model like SkipGramEmbedding or CBOWEmbedding
  protected def process(ep: Int, rel: Int): Unit
  def getScore(ep: Int, rel: Int): Double
}