package cc.factorie.app.nlp.embeddings.transRelations

import java.io.{File, PrintWriter, FileInputStream}
import java.util
import java.util.zip.GZIPInputStream

import cc.factorie.app.nlp.embeddings.{Evaluator, UniversalSchemaModel, EmbeddingOpts}
import cc.factorie.model.{Parameters, Weights}
import cc.factorie.optimize._
import cc.factorie.util.CmdOptions

import scala.collection.mutable
import scala.collection.mutable.ArrayBuffer
import scala.util.Random

/**
 * Created by pat on 4/3/15.
 */
abstract class TransRelationModel(override val opts: EmbeddingOpts) extends UniversalSchemaModel(opts) {

  val gamma = opts.margin.value
  // use L1 distance, L2 otherwise
  val l1 = if (opts.l1.value) true else false

  // throw out relations occuring less than min relation count times
  val minRelationCount = 1
  val negativeSamples = 1
  val bernoulliSample = opts.bernoulliSample.value

//  var trainTriplets = Seq[(String, String, String)]()


  protected val iterations = opts.epochs.value
  protected val batchSize = opts.batchSize.value


  protected var relationBernoulli : Map[Int, Double] = null
//  protected val relationVocab = new util.HashMap[String, Int]
//  var relationCount = 0



//  override  def buildVocab(): Unit ={
//    println("Building Vocab")
//    val relationMap = readInFile(corpus)
//    relationBernoulli = calculateRelationBernoulli(relationMap.toSeq)
//    // flatten input triplets
//    trainTriplets = relationMap.filter(eList => eList._2.size >= minRelationCount).toSeq.flatMap(eList => eList._2.toSet.toSeq)
//  }

//  override def evaluate(file: String, iter: Int): Double = {
//
//    val  corpusLineItr = io.Source.fromInputStream(new FileInputStream(file), encoding).getLines
//    val ans = scala.collection.mutable.Map[Int, ArrayBuffer[(Double, Boolean)]]()
//
//    val fileName = file + "_" + "output" +  "_" + D.toString + "_" + adaGradRate.toString + "_" + opts.regularizer.value.toString + "_" + opts.negative.value.toString + "_" + iter.toString + "_"  + opts.hinge.value.toString + "_" + opts.margin.value.toString + "_" + opts.options.value.toString
//    println(fileName)
//    val p = if(opts.writeOutput.value)  new PrintWriter(new File(fileName))
//    try{
//      while (corpusLineItr.hasNext) {
//        val line = corpusLineItr.next
//        val Array(ep, rel, label) = line.stripLineEnd.split('\t')
//        var truth = true
//        if(label == "0")  truth = false
//        val s = getScore(ep, rel)
//        if(opts.writeOutput.value)  p.asInstanceOf[PrintWriter].write(rel + "\t0\t" + ep + "\t0\t" + s.toString + "\tmycode\n")
//        ans(relationVocab.get(rel)) = ans.getOrElseUpdate(relationVocab.get(rel), ArrayBuffer[(Double, Boolean)]()) += ((s,truth))
//      }
//    }
//    finally {
//      if(opts.writeOutput.value)  p.asInstanceOf[PrintWriter].close()
//    }
//    var predictionSize = 0
//    for(i <- ans.keys) {
//      ans(i) = rand.shuffle(ans(i))
//      predictionSize += ans(i).size
//    }
//    println("prediction size : ", predictionSize)
//    Evaluator.meanAveragePrecision(ans)
//  }

  def readInFile(inFile : String): Map[String, ArrayBuffer[(String, String, String)]] ={
    val corpusLineItr = inFile.endsWith(".gz") match {
      case true => io.Source.fromInputStream(new GZIPInputStream(new FileInputStream(inFile)), encoding).getLines()
      case false => io.Source.fromInputStream(new FileInputStream(inFile), encoding).getLines()
    }
    val relationMap = new mutable.HashMap[String, ArrayBuffer[(String, String, String)]]
    while (corpusLineItr.hasNext) {
      val line = corpusLineItr.next()
      val (entPair, e1, e2, relation, score) = if (opts.parseTsv.value) parseTsv(line) else parseArvind(line)
//      if (!entityVocab.containsKey(e1)) {
//        entityVocab.put(e1, entityCount)
//        entityCount += 1
//      }
//      if (!entityVocab.containsKey(e2)) {
//        entityVocab.put(e2, entityCount)
//        entityCount += 1
//      }
//      if (!relationVocab.containsKey(relation)) {
//        relationVocab.put(relation, relationCount)
//        relationCount += 1
//      }
      relationMap.put(relation, relationMap.getOrElse(relation, new ArrayBuffer()) += ((e1, relation, e2)))
    }
    relationMap.toMap
  }

  /**
   * Calcualte the distribution of unique head and tails for each relation for bernouli negative sampling
   * @param relationTripleMap map from relation to triples containing that relation
   * @return map from relation index to P(corrupt head)
   */
  def calculateRelationBernoulli(relationTripleMap : Seq[(String, ArrayBuffer[(String, String, String)])]): Map[Int, Double] ={
    relationTripleMap.map { case (rel, triples) =>
      // group by head entity
      val headGroups = triples.groupBy(_._1)
      // distinct tails per head
      val tph = (headGroups.flatMap(triple => triple._2.map(_._3)).size / headGroups.size).toDouble
      // group by tail entity
      val tailGroups = triples.groupBy(_._3)
      // distinct heads per tail
      val hpt = (tailGroups.flatMap(triple => triple._2.map(_._1)).size / tailGroups.size).toDouble
      relationKey.get(rel) -> (tph / (tph + hpt))
    }.toMap
  }

  /**
   * normalizes weights to be either exactly 1 or <= 1
   * @param embeddings the weights to be normalized
   * @param exactlyOne if true, normalize all weights to be length 1, otherwise enforce lengths <= 1
   */
  def normalize(embeddings : Seq[Weights], exactlyOne:Boolean = true): Unit ={
    embeddings.par.foreach(e => {
      val vec = e.value
      val len = vec.twoNorm
      if (exactlyOne || len > 1.0)
        vec /= len
    })
  }

  protected def generateMiniBatch(trainTriplets: List[(Int, Int, Int, Int)], batchSize: Int): Seq[Example] = {
//    Seq.fill(batchSize)(trainTriplets(rand.nextInt(trainTriplets.size))).map { case (ePair, e1, e2, rel) =>
    trainTriplets.map { case (ePair, e1, e2, rel) =>
      makeExample(e1, rel + entityCount, e2)
    }
  }

  def makeExample(e1Index : Int, relationIndex : Int, e2Index : Int): Example

  /**
   * Randomly sample an entity
   * @param exclude A set of entities to exclude
   * @return an entity's id
   */
  def negativeSampleEntity(exclude : Option[Set[String]]): Int ={
    var negIndex = rand.nextInt(entityCount)
    if (exclude != None) {
      val excludeIndices = exclude.get.map(entityVocab.get)
      while (excludeIndices.contains(negIndex)) negIndex = rand.nextInt(entityCount)
    }
    negIndex
  }

  def negativeSample(e1Dex : Int, e2Dex : Int, relDex : Int, exclude : Option[Set[String]]): (Int, Int) =
  {
    val sampleTail = if(bernoulliSample && relationBernoulli != null)
      rand.nextDouble() >= relationBernoulli.getOrElse(relDex, 0.5)
    else rand.nextInt(2) == 0
    if (sampleTail) (e1Dex, negativeSampleEntity(None)) else (negativeSampleEntity(None), e2Dex)
  }

  def getScore(triple : (String, String, String)) : Double
}


