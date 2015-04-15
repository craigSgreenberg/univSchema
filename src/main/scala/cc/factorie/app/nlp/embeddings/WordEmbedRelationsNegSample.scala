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
  var wordEmbeddings : Seq[Weights] = null
  val wordVocab = new util.HashMap[String, Int]
  var wordEmbedVocabSize = 0
  var wordEmbedD = 0

  override def process(ep: Int, rel: Int): Unit = {
    trainer.processExample(new WordEmbedRelationNegSampleExample(this, ep, rel))
  }


  override def buildVocab(): Unit ={
    super.buildVocab()
    val st1 = System.currentTimeMillis()
    if (opts.loadVocabFile.hasValue) loadWordEmbeddings(opts.loadVocabFile.value)
    else randomlyInitializeWordEmbeddings()
    println(s"time taken to load $wordEmbedVocabSize embeddings : ${(System.currentTimeMillis() - st1) / 1000.0}")
    println("# words : %d , # size : %d".format(wordEmbedVocabSize, wordEmbedD))
  }

  /**
   * Randomly initialize a word embedding for each unique whitespace
   * seperated token appearing across relations
   */
  def randomlyInitializeWordEmbeddings(): Unit =
  {
    val relWords = relationKey.keysIterator.flatMap(_.split("\\s+")).toSet
    wordEmbedD = D
    wordEmbedVocabSize = relWords.size
    vocab = new Array[String](wordEmbedVocabSize)
    wordEmbeddings = relWords.zipWithIndex.map{case(word, i) =>
      vocab(i) = word
      wordVocab.put(word, i)
      Weights(TensorUtils.setToRandom1(new DenseTensor1(wordEmbedD, 0), rand))
    }
  }

  /**
   * load word embeddings word2vec formated txt file
   * @param embeddingsFile path to word2vec.txt file
   * @param encoding encoding
   */
  def loadWordEmbeddings(embeddingsFile: String, encoding: String = "UTF8"): Unit =
  {
    val lineItr = Source.fromFile(embeddingsFile, encoding).getLines()
    // first line is (# words, dimension)
    val details = lineItr.next.stripLineEnd.split(' ').map(_.toInt)
    wordEmbedVocabSize = if (threshold > 0 && details(0) > threshold) threshold else details(0)
    wordEmbedD = details(1)
    vocab = new Array[String](wordEmbedVocabSize)
    wordEmbeddings = (0 until wordEmbedVocabSize).map(i => {
      val line = lineItr.next.stripLineEnd.split(' ')
      val word = line(0)
      vocab(i) = word
      wordVocab.put(word, i)
      val v = new DenseTensor1(wordEmbedD, 0) // allocate the memory
      for (d <- 0 until wordEmbedD) v(d) = line(d + 1).toDouble
      v.twoNormalize()
      Weights(v)
    })
  }


  override def getScore(ep: Int, rel: Int): Double = {
    val nodeEmbedding = aggregateWordEmbeddings(reverseRelationKey(rel))._1
    val epEmbedding = weights(ep).value
    val ans: Double = nodeEmbedding.dot(epEmbedding)
    ans
  }


  def aggregateWordEmbeddings(relStr : String) : (DenseTensor1, Array[Int]) =
  {
    val relationEmbedding = new DenseTensor1(wordEmbedD, 0)
    val wordIds = relStr.split("\\s+").map(word => if(wordVocab.containsKey(word)) wordVocab.get(word) else -1).filterNot(_ == -1)
    val wordVectors = wordIds.map(wordEmbeddings(_).value)
    wordVectors.foreach(v => relationEmbedding.+=(v))
    relationEmbedding./=(Math.max(1.0, wordIds.length))
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
