package cc.factorie.app.nlp.embeddings

import scala.collection.mutable
import cc.factorie.la.DenseTensor1

/*
class CBOWHSoftMax (override val opts: EmbeddingOpts) extends WordEmbeddingModel(opts) {
  val negative = opts.negative.value
  val window = opts.window.value
  val rng = new util.Random(5) // fix the speed;
  val sample = opts.sample.value.toDouble

  override def process(doc: String): Int = {
    // given a document, below line splits by space and converts each word to Int (by vocab.getId) and filters out words not in vocab
    var sen = doc.stripLineEnd.split(' ').map(word => vocab.getId(word)).filter(id => id != -1)
    val wordCount = sen.size

    // subsampling -> speed increase
    if (sample > 0)
      sen = sen.filter(id => subSample(id) != -1)

    val senLength = sen.size
    for (senPosition <- 0 until senLength) {
      val currWord = sen(senPosition)
      val b = rng.nextInt(window)
      val contexts = new mutable.ArrayBuffer[Int]
      // make the contexts
      for (a <- b until window * 2 + 1 - b) if (a != window) {
        val c = senPosition - window + a
        if (c >= 0 && c < senLength)
          contexts += sen(c)
      }
      // make the examples
      trainer.processExample(new CBOWHierarchicalSoftMax(this, currWord, contexts, vocab.getCodelength(currWord),vocab.getCode(currWord),vocab.getPoint(currWord)))

    }
    return wordCount
  }
  // subsampling
  def subSample(word: Int): Int = {
    val prob = vocab.getSubSampleProb(word) // pre-computed to avoid sqrt call every time. Improvement of 10 secs on 100MB data ~ 15 MINs on 10GB
    val alpha = rng.nextInt(0xFFFF) / 0xFFFF.toDouble
    if (prob < alpha) { return -1 }
    else return word
  }
}

class CBOWHierarchicalSoftMax(model: WordEmbeddingModel, word: Int, wordContexts: Seq[Int], codeLen:Int,code: Array[Int],point:Array[Int]) extends Example {
  def accumulateValueAndGradient(value: DoubleAccumulator, gradient: WeightsMapAccumulator): Unit = {

    val contextEmbedding = new DenseTensor1(model.D, 0)
    wordContexts.foreach(context => contextEmbedding.+=(model.weights(context).value))




    for (d <- 0 until codeLen) {
      val currNode = point(d)
      val nodeEmbedding = model.nodeWeights(currNode).value
      val score: Double = nodeEmbedding.dot(contextEmbedding)
      val exp: Double = math.exp(-score)

      var objective: Double = 0.0
      var factor: Double = 0.0
      if (code(d) == 1) {
        objective = -math.log1p(exp)
        factor = exp / (1 + exp)
      }
      if (code(d) == 0) {
        objective = -score - math.log1p(exp)
        factor = -1 / (1 + exp)
      }
      if (value ne null) value.accumulate(objective)
      if (gradient ne null) {
        wordContexts.foreach(context => gradient.accumulate(model.weights(context), nodeEmbedding, factor))
        gradient.accumulate(model.nodeWeights(currNode), contextEmbedding, factor)
      }

    }
  }
}
*/

