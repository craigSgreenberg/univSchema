package cc.factorie.app.nlp.embeddings

import java.util

import cc.factorie.la.{DenseTensor1, WeightsMapAccumulator}
import cc.factorie.optimize.Example
import cc.factorie.util.DoubleAccumulator

import scala.io.Source

/**
 * Created by pat on 4/15/15.
 */
class WordEmbedRelationsNegSample(override val opts: EmbeddingOpts) extends UniversalSchemaModel(opts) {
  var threshold = 0
  var vocab = Array[String]()
  var relationEmbeddings = Array[DenseTensor1]()
  var wordEmbeddings = Array[DenseTensor1]()
  val wordVocab = new util.HashMap[String, Int]
  var zeroTensor : DenseTensor1 = null
  var wordEmbedVocabSize = 0
  var wordEmbedD = 0

  loadWordEmbeddings("/home/pat/data/GoogleNews-vectors-negative300.txt")
  wordSumRelationEmbeddings()

  override def process(ep: Int, rel: Int): Unit = {
    trainer.processExample(new WordEmbedRelationNegSampleExample(this, ep, rel))
  }

  override def getScore(ep: Int, rel: Int): Double = {
    val nodeEmbedding = relationEmbeddings(rel)
    val epEmbedding = weights(ep).value
    val ans: Double = nodeEmbedding.dot(epEmbedding)
    ans
  }

  def subSample(word: Int): Int = {
    -1
  }

  def loadWordEmbeddings(embeddingsFile: String, encoding: String = "UTF8"): Unit = {
    val lineItr = Source.fromFile(embeddingsFile, encoding).getLines
    // first line is (# words, dimension)
    val details = lineItr.next.stripLineEnd.split(' ').map(_.toInt)
    wordEmbedVocabSize = if (threshold > 0 && details(0) > threshold) threshold else details(0)
    wordEmbedD = details(1)
    println("# words : %d , # size : %d".format(wordEmbedVocabSize, wordEmbedD))
    vocab = new Array[String](wordEmbedVocabSize)
    wordEmbeddings = new Array[DenseTensor1](wordEmbedVocabSize)
    for (v <- 0 until wordEmbedVocabSize) {
      val line = lineItr.next.stripLineEnd.split(' ')
      val word = line(0).toLowerCase
      vocab(v) = word
      wordVocab.put(word, v)
      wordEmbeddings(v) = new DenseTensor1(wordEmbedD, 0) // allocate the memory
      for (d <- 0 until wordEmbedD) wordEmbeddings(v)(d) = line(d + 1).toDouble
      wordEmbeddings(v) /= wordEmbeddings(v).twoNorm
    }
    zeroTensor = new DenseTensor1(wordEmbedD, 0)
    println("loaded vocab and their embeddings")
  }

  def wordSumRelationEmbeddings()
  {
    relationEmbeddings = new Array[DenseTensor1](relationKey.size)
    relationKey.foreach{ case (relStr, relKey) =>
        relationEmbeddings(relKey) = aggregateWordEmbeddings(relStr)
    }
  }

  def aggregateWordEmbeddings(relStr : String) : DenseTensor1 =
  {
    val relationEmbedding = new DenseTensor1(D, 0)
    val wordIds = relStr.split("\\s+").map(word => wordVocab.getOrDefault(word, -1)).filterNot(_ == -1)
    wordIds.foreach(wordId => relationEmbedding.+=(wordEmbeddings(wordId)))
    relationEmbedding./=(Math.max(1, wordIds.length))
    relationEmbedding
  }

}


class WordEmbedRelationNegSampleExample(model: WordEmbedRelationsNegSample, ep: Int, rel: Int) extends Example {
  val attempts = 20

  def getNegEp(): Int = {
    var trial = 0
    var found = false
    var ret = -1
    while (trial < attempts && (!found)) {
      val neg = model.rand.nextInt(model.entPairSize)
      if (!(model.positives(rel).contains(neg))) {
        found = true
        ret = neg
      }
      trial += 1
    }
    ret
  }

  def accumulateValueAndGradient(value: DoubleAccumulator, gradient: WeightsMapAccumulator): Unit =
  {
    for (j <- 0 until model.opts.negative.value) {
      var negEp = getNegEp()
      if (negEp == -1) return

      model.processed += 1
      val nodeEmbedding = model.relationEmbeddings(rel)
      val epEmbedding = model.weights(ep)
      val negEpEmbedding = model.weights(negEp)

      val diff: Double = model.getScore(ep, rel) - model.getScore(negEp, rel)
      val objective: Double = 0.0
      var factor: Double = 1 / (1 + math.exp(-diff))
      factor = 1 - factor
      if (value ne null) value.accumulate(objective)
      if (gradient ne null) {
        gradient.accumulate(epEmbedding, nodeEmbedding, factor)
        gradient.accumulate(negEpEmbedding, nodeEmbedding, -1 * factor)
//        gradient.accumulate(model.nodeWeights(rel), (epEmbedding - negEpEmbedding), factor)
      }

    }
  }
}
