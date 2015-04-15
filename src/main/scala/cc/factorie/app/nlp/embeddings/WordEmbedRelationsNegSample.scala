package cc.factorie.app.nlp.embeddings

import java.util

import cc.factorie.la.{DenseTensor1, WeightsMapAccumulator}
import cc.factorie.model.Weights
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
  var wordEmbeddings : Seq[Weights] = null
  val wordVocab = new util.HashMap[String, Int]
  var zeroTensor : DenseTensor1 = null
  var wordEmbedVocabSize = 0
  var wordEmbedD = 0

  if (opts.loadVocabFile.hasValue) loadWordEmbeddings(opts.loadVocabFile.value)

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
    val lineItr = Source.fromFile(embeddingsFile, encoding).getLines()
    // first line is (# words, dimension)
    val details = lineItr.next.stripLineEnd.split(' ').map(_.toInt)
    wordEmbedVocabSize = if (threshold > 0 && details(0) > threshold) threshold else details(0)
    wordEmbedD = details(1)
    println("# words : %d , # size : %d".format(wordEmbedVocabSize, wordEmbedD))
    vocab = new Array[String](wordEmbedVocabSize)
    wordEmbeddings = (0 until wordEmbedVocabSize).map(i => {
      val line = lineItr.next.stripLineEnd.split(' ')
      val word = line(0).toLowerCase
      vocab(i) = word
      wordVocab.put(word, i)
      val v = new DenseTensor1(wordEmbedD, 0) // allocate the memory
      for (d <- 0 until wordEmbedD) v(d) = line(d + 1).toDouble
      v.twoNormalize()
      Weights(v)
    })
    zeroTensor = new DenseTensor1(wordEmbedD, 0)
    println("loaded vocab and their embeddings")
  }

  def aggregateWordEmbeddings(relStr : String) : (DenseTensor1, Array[Int]) =
  {
    val relationEmbedding = new DenseTensor1(D, 0)
    val wordIds = relStr.split("\\s+").map(word => wordVocab.getOrDefault(word, -1)).filterNot(_ == -1)
    val wordVectors = wordIds.map(wordEmbeddings(_))
    wordVectors.foreach(v => relationEmbedding.+=(v.value))
    relationEmbedding./=(Math.max(1, wordIds.length))
    (relationEmbedding, wordIds)
  }

}


class WordEmbedRelationNegSampleExample(model: WordEmbedRelationsNegSample, ep: Int, rel: Int) extends NegativeSamplingExample(model, ep, rel) {

  override def accumulateValueAndGradient(value: DoubleAccumulator, gradient: WeightsMapAccumulator): Unit =
  {
    for (j <- 0 until model.opts.negative.value) {
      val negEp = getNegEp()
      if (negEp == -1) return

      model.processed += 1
      val relString = model.vocab(rel)//model.relationEmbeddings(rel)
      val (relEmbedding, relWordIds) = model.aggregateWordEmbeddings(relString)
      val epEmbedding = model.weights(ep).value
      val negEpEmbedding = model.weights(negEp).value

      val diff: Double = model.getScore(ep, rel) - model.getScore(negEp, rel)
      val objective: Double = 0.0
      var factor: Double = 1 / (1 + math.exp(-diff))
      factor = 1 - factor
      if (value ne null) value.accumulate(objective)
      if (gradient ne null) {
        gradient.accumulate(model.weights(ep), relEmbedding, factor)
        gradient.accumulate(model.weights(negEp), relEmbedding, -factor)
        relWordIds.foreach(v => {
          gradient.accumulate(model.wordEmbeddings(rel), epEmbedding - negEpEmbedding, factor)
        })
      }

    }
  }
}
