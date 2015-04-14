package cc.factorie.app.nlp.embeddings.transRelations

import java.io.FileInputStream
import java.util
import java.util.zip.GZIPInputStream

import cc.factorie.model.{Parameters, Weights}
import cc.factorie.optimize._
import cc.factorie.util.CmdOptions

import scala.collection.mutable
import scala.collection.mutable.ArrayBuffer
import scala.util.Random

/**
 * Created by pat on 4/3/15.
 */
abstract class TransRelationModel(val opts: TransRelationOpts) extends Parameters {

  val D = opts.dimension.value
  var weights: Seq[Weights] = null
  val gamma = opts.gamma.value
  // use L1 distance, L2 otherwise
  val l1 = if (opts.l1.value) true else false

  // throw out relations occuring less than min relation count times
  val minRelationCount = 1
  val negativeSamples = 1

  val threads = opts.threads.value
  val adaGradDelta = 0.0
  val adaGradRate = opts.rate.value
  val encoding = "UTF-8"

  protected val iterations = opts.iterations.value
  protected val batchSize = opts.batchSize.value

  protected var trainer: Trainer = null
  protected var optimizer: GradientOptimizer = null

  protected val relationVocab = new util.HashMap[String, Int]
  protected val entityVocab = new util.HashMap[String, Int]
  var relationCount = 0
  var entityCount = 0

  val rand = new Random(69)


  def buildVocab(inFile: String, lineParser: String => (String, String, String)): Seq[(String, String, String)] = {
    println("Building Vocab")
    val corpusLineItr = inFile.endsWith(".gz") match {
      case true => io.Source.fromInputStream(new GZIPInputStream(new FileInputStream(inFile)), encoding).getLines()
      case false => io.Source.fromInputStream(new FileInputStream(inFile), encoding).getLines()
    }
    val relationMap = new mutable.HashMap[String, ArrayBuffer[(String, String, String)]]
    while (corpusLineItr.hasNext) {
      val line = corpusLineItr.next()
      val (e1, relation, e2) = lineParser(line)
      if (!entityVocab.containsKey(e1)) {
        entityVocab.put(e1, entityCount)
        entityCount += 1
      }
      if (!entityVocab.containsKey(e2)) {
        entityVocab.put(e2, entityCount)
        entityCount += 1
      }
      if (!relationVocab.containsKey(relation)) {
        relationVocab.put(relation, relationCount)
        relationCount += 1
      }
      relationMap.put(relation, relationMap.getOrElse(relation, new ArrayBuffer()) += ((e1, relation, e2)))
    }
    // flatten input triplets
    relationMap.filter(eList => eList._2.size >= minRelationCount).toSeq.flatMap(eList => eList._2.toSet.toSeq)
  }

  // assumes arvind format : [e1,e2\trelation\tscore]
  def parseArvind(line: String): (String, String, String) = {
    val Array(entities, relation, score) = line.split("\t")
    val Array(e1, e2) = entities.split(",")
    (e1, relation, e2)
  }

  def parseTsv(line: String): (String, String, String) = {
    val parts = line.split("\t")
    (parts(0), parts(1), parts(2))
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

  protected def generateMiniBatch(trainTriplets: Seq[(String, String, String)], batchSize: Int): Seq[Example] = {
    Seq.fill(batchSize)(trainTriplets(rand.nextInt(trainTriplets.size))).map { case (e1, relation, e2) =>
      val e1Index = entityVocab.get(e1)
      val e2Index = entityVocab.get(e2)
      val relationIndex = relationVocab.get(relation) + entityCount
      // corrupt either head or tail
      makeExample(e1Index, relationIndex, e2Index)
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

  def getScore(triple : (String, String, String)) : Double
}


class TransRelationOpts extends CmdOptions {
  val train = new CmdOption[String]("train", "", "FILENAME", "Train file.")
  val test = new CmdOption[String]("test", "", "FILENAME", "Test File.")
  val l1 = new CmdOption[Boolean]("l1", true, "BOOLEAN", "Use l1 distance, l2 otherwise")
  val iterations = new CmdOption[Int]("iterations", 10, "INT", "Number of iterations to run.")
  val threads = new CmdOption[Int]("threads", 20, "INT", "Number of iterations to run.")
  val dimension = new CmdOption[Int]("dimension", 100, "INT", "Number of iterations to run.")
  val batchSize = new CmdOption[Int]("batch-size", 1200, "INT", "Size of each mini batch")
  val rate = new CmdOption[Double]("rate", 0.01, "INT", "Learning rate.")
  val gamma = new CmdOption[Double]("gamma", 1.0, "DOUBLE", "Number of mini batches to use.")
}


