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
  val bernoulliSample = false // TODO broken in the refactor opts.bernoulliSample.value


  protected val iterations = opts.epochs.value
  protected val batchSize = opts.batchSize.value
  protected var relationBernoulli : Map[Int, Double] = null


  def fileToTriplets(inFile : String): Map[String, ArrayBuffer[(String, String, String)]] ={
    val corpusLineItr = inFile.endsWith(".gz") match {
      case true => io.Source.fromInputStream(new GZIPInputStream(new FileInputStream(inFile)), encoding).getLines()
      case false => io.Source.fromInputStream(new FileInputStream(inFile), encoding).getLines()
    }
    val relationMap = new mutable.HashMap[String, ArrayBuffer[(String, String, String)]]
    while (corpusLineItr.hasNext) {
      val line = corpusLineItr.next()
      val (entPair, e1, e2, relation, score) = if (opts.parseTsv.value) parseTsv(line) else parseArvind(line)
      relationMap.put(relation, relationMap.getOrElse(relation, new ArrayBuffer()) += ((e1, relation, e2)))
    }
    relationMap.toMap
  }

  /**
   * Calculate the distribution of unique head and tails for each relation for bernouli negative sampling
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

  protected def generateMiniBatch(): Seq[Example] = {
    Seq.fill(batchSize)(trainingExamples(rand.nextInt(trainingExamples.size))).map { case (ePair, e1, e2, rel) =>
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


