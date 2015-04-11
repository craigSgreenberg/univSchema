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
import cc.factorie.model.{ Parameters, Weights }
import cc.factorie.optimize.{HogwildTrainer, Trainer, AdaGradRDA}
import cc.factorie.la.{SparseHashTensor1, SparseBinaryTensor1, DenseTensor1}
import cc.factorie.util.Threading
import java.io._
import java.util.zip.{ GZIPOutputStream, GZIPInputStream }
import scala.collection.mutable
import scala.util.control.Breaks._
import scala.util.Random
import scala.collection.mutable.ArrayBuffer

abstract class WordEmbeddingModel(val opts: EmbeddingOpts) extends Parameters {
  //val entityPairFeatures = new mutable.HashMap[Int, ArrayBuffer[Int]]()
  val entityPairFeatures = new mutable.HashMap[Int, SparseBinaryTensor1]()
  var classifierWeights: Seq[Weights] = null
  val testRels: mutable.HashSet[String] = new mutable.HashSet[String]()
  var processed = 0
  // Algo related
  val rand = new Random(0)
  val D = opts.dimension.value // default value is 200
  var entPairSize: Int = -1 // no. of entity pairs
  var relationSize: Int = -1 // no. of relations
  var trainingExamplesSize : Int = -1 //no. of training examples
  var tree = new mutable.HashMap[Int, treePosition]()
  val positives = new mutable.HashMap[Int, mutable.HashSet[Int]]()
  protected val entPairKey = new mutable.HashMap[String, Int]()
  protected val relationKey = new mutable.HashMap[String, Int]()
  protected val reverseRelationKey = new mutable.HashMap[Int, String]()
  protected var trainingExamples = List[(Int, Int)]()
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

  def meanAveragePrecision(classToPredictionAndLabel: scala.collection.mutable.Map[Int, ArrayBuffer[(Double, Boolean)]]): Double = {
    classToPredictionAndLabel.values.map(averagePrecision(_)).sum / classToPredictionAndLabel.size.toDouble
  }

  def averagePrecision(predictionAndLabel: ArrayBuffer[(Double, Boolean)]): Double = {
    val judgements = predictionAndLabel.sortBy(-_._1).map(_._2)
    // This foldleft aggregates (#true, #false, averagePrecision).
    // #true and #false are simply counted up.
    // averagePrecision (ap) is updated for true items:
    // ap_new = (#true_new - 1) * ap_old / #true_new + 1 / (#true_new + #false_new)
    //        = (#true_old) * ap_old / (#true_old + 1) + 1 / (#true_old + #false_old + 1)
    val mapStats = judgements.foldLeft((0,0,0.0))((stat,j) => {if (j==true) {
      (stat._1 + 1, stat._2, stat._1 * stat._3 / (stat._1 + 1) + 1.0/(stat._1 + stat._2 + 1))
    } else {
      (stat._1, stat._2 + 1, stat._3)
    }})
    mapStats._3
  }

  // convenience method
  def averagePrecision(classToPredictionAndLabel: scala.collection.mutable.Map[Int, Seq[(Double, Boolean)]]): Double = {
    throw new UnsupportedOperationException
  }

  def buildBinaryTree(){
    val treeFile =   io.Source.fromInputStream(new FileInputStream(opts.treeFile.value), encoding).getLines
    while (treeFile.hasNext) {
     val Array(rel, nl, code) =  (treeFile.next).split("\t")
      tree.getOrElseUpdate(relationKey(rel), new treePosition(rel, nl.split(" ").toSeq.map(c => c.toInt).toArray, code.split(" ").toSeq.map(c => c.toInt).toArray))
    }
  }

  def buildVocab(): Unit ={
    val corpusLineItr = corpus.endsWith(".gz") match {
      case true => io.Source.fromInputStream(new GZIPInputStream(new FileInputStream(corpus)), encoding).getLines
      case false => io.Source.fromInputStream(new FileInputStream(corpus), encoding).getLines
    }
    while (corpusLineItr.hasNext) {
      val line = corpusLineItr.next
      val Array(ep, rel, label) = line.stripLineEnd.split('\t')
      trainingExamples = (entPairKey.getOrElseUpdate(ep, entPairKey.size), relationKey.getOrElseUpdate(rel, relationKey.size)) :: trainingExamples
      positives(relationKey(rel)) = positives.getOrElseUpdate(relationKey(rel), mutable.HashSet[Int]())
      positives(relationKey(rel)).add(entPairKey(ep))
      reverseRelationKey(relationKey(rel)) = rel
    }

    entPairSize = entPairKey.size
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
    println("Number of entity pairs: ", entPairSize)
    println("Number of relations: ", relationSize, positives.size, reverseRelationKey.size)
    println("Number of training examples: ", trainingExamplesSize)
    if(opts.options.value == 1)  buildBinaryTree()
  }

  def evaluate(file: String, iter: Int): Double = {
    val  corpusLineItr = io.Source.fromInputStream(new FileInputStream(file), encoding).getLines
    val ans = scala.collection.mutable.Map[Int, ArrayBuffer[(Double, Boolean)]]()
    //val p = if(opts.writeOutput.value)  new PrintWriter(new File(file + "_" + "output" + "_" + D.toString + "_" + adaGradRate.toString + "_" + opts.regularizer.value.toString + "_" + opts.negative.value.toString + "_" + iter.toString + "_"  + opts.hinge.value.toString + "_" + opts.wsabie.value.toString + "_" + opts.margin.value.toString))
    val fileName = file + "_" + "output" +  "_" + D.toString + "_" + adaGradRate.toString + "_" + opts.regularizer.value.toString + "_" + opts.negative.value.toString + "_" + iter.toString + "_"  + opts.hinge.value.toString + "_" + opts.wsabie.value.toString + "_" + opts.margin.value.toString + "_" + opts.treeFile.value.toString.split("/").reverse(0)
    println(fileName)
    val p = if(opts.writeOutput.value)  new PrintWriter(new File(fileName))
    try{
      while (corpusLineItr.hasNext) {
        val line = corpusLineItr.next
        val Array(ep, rel, label) = line.stripLineEnd.split('\t')
        var truth = true
        if(label == "0")  truth = false
        val s = getScore(entPairKey(ep), relationKey(rel))
        if(opts.writeOutput.value)  p.asInstanceOf[PrintWriter].write(rel + "\t0\t" + ep + "\t0\t" + s.toString + "\tmycode\n")
        //if(truth) k.write(rel + " 0 " + ep + " 1\n")
        ans(relationKey(rel)) = ans.getOrElseUpdate(relationKey(rel), ArrayBuffer[(Double, Boolean)]()) += ((s,truth))
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
    meanAveragePrecision(ans)
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
          val it = rand.shuffle(trainingExamples).grouped(groupSize);
          //println("match ", threads, groupSize)
          var threadExamples = new ArrayBuffer[List[(Int, Int)]]()
          for(n <- 0 until threads)  threadExamples = threadExamples += rand.shuffle(it.next())
          Threading.parForeach(threadIds, threads)(threadId => workerThread(threadExamples(threadId)))
          if(i % opts.evalautionFrequency.value == 0) {
            println("Dev MAP after " + i + " iterations: " + evaluate(opts.devFile.value, i))
            println("Test MAP after " + i + " iterations: " + evaluate(opts.testFile.value, i))
          }
    }
    println("Done learning embeddings. ")
    val out = if(opts.writeVecs.value)  new PrintWriter(new File("relationVecs"))
    try{
        if(opts.writeVecs.value){
          for(i <- 0 until relationSize)  out.asInstanceOf[PrintWriter].write(reverseRelationKey(i) + "\t" + nodeWeights(i).value.toString + "\n")
        }
    }
    finally {
      if(opts.writeVecs.value) out.asInstanceOf[PrintWriter].close()
    }
  }

  protected def workerThread(examples: List[(Int, Int)]): Unit = {
    //println("processing: " + examples.size)
    examples.foreach(example => process(example._1, example._2))
  }

  // override this function in your Embedding Model like SkipGramEmbedding or CBOWEmbedding
  protected def process(ep: Int, rel: Int): Unit
  def getScore(ep: Int, rel: Int): Double
}

class treePosition(w: String, nl: Array[Int]=new Array[Int](40), cd: Array[Int] = new Array[Int](40)) {
  var relation = w
  var code = cd // 0's and 1's indicating directions
  var codelen = cd.size
  var nodeList = nl //list of nodes in the path
  override def toString() = "( "  + relation +  " ) "
}