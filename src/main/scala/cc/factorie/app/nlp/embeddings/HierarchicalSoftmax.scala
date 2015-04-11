package cc.factorie.app.nlp.embeddings

import scala.collection.mutable
import cc.factorie.optimize.Example
import cc.factorie.util.DoubleAccumulator
import cc.factorie.la.{DenseTensor1, WeightsMapAccumulator}


class HierarchicalSoftmax (override val opts: EmbeddingOpts) extends UniversalSchemaModel(opts) {
  override def process(ep: Int, rel: Int): Unit = {
        trainer.processExample(new HierarchicalSoftMaxExample(this, ep, tree(rel).codelen, tree(rel).code, tree(rel).nodeList))
  }
  override def getScore(ep: Int, rel: Int): Double = {
    //trainer.processExample(new ParagraphVectorHierarchicalSoftMaxExample(this, entPairKey(ep), tree(r).codelen, tree(r).code, tree(r).nodeList))
    var ans = 0.0
    for (d <- 0 until tree(rel).codelen) {
      val currNode = tree(rel).nodeList(d)
      //println(currNode)
      val nodeEmbedding = nodeWeights(currNode).value
      val epEmbedding =   weights(ep).value
      val score: Double = nodeEmbedding.dot(epEmbedding)
      val exp: Double = 1/(1 + math.exp(-score))
      if(exp == 0.0)  println("error!!!!!!!!!!")
      if(exp > 1000000) println("error!!!!!!!!!!")
      if (tree(rel).code(d) == 1) ans = ans  + math.log(exp)
      else if (tree(rel).code(d) == 0) ans = ans  + math.log((1 - exp))
    }
    ans
  }

  def subSample(word: Int): Int = {
    /*
    val prob = vocab.getSubSampleProb(word) // pre-computed to avoid sqrt call every time.
    val alpha = rng.nextInt(0xFFFF) / 0xFFFF.toDouble
    if (prob < alpha) { return -1 }
    else return word
    */
    return -1
  }
}

/*class ParagraphVectorNegativeSampling(model: WordEmbeddingModel, word: Int, wordContexts: Seq[Int],parContext: Int) extends Example {
  def accumulateValueAndGradient(value: DoubleAccumulator, gradient: WeightsMapAccumulator): Unit = {

    val wordEmbedding = model.weights(word).value
    val contextEmbedding = new DenseTensor1(model.D, 0)
    wordContexts.foreach(context => contextEmbedding.+=(model.weights(context).value))
    contextEmbedding.+=(model.parWeights(parContext).value)
    val score: Double = wordEmbedding.dot(contextEmbedding)
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


  }
}  */



class HierarchicalSoftMaxExample(model: UniversalSchemaModel, ep: Int, codeLen:Int, code: Array[Int],nodeList:Array[Int]) extends Example {
  def accumulateValueAndGradient(value: DoubleAccumulator, gradient: WeightsMapAccumulator): Unit = {
    /*
    val contextEmbedding = new DenseTensor1(model.D, 0)
    wordContexts.foreach(context => contextEmbedding.+=(model.weights(context).value))
    contextEmbedding.+=(model.parWeights(parContext).value)
    */
    model.processed += 1
    for (d <- 0 until codeLen) {
      val currNode = nodeList(d)
      //println(currNode)
      val nodeEmbedding = model.nodeWeights(currNode).value
      val epEmbedding =   model.weights(ep).value
      val score: Double = nodeEmbedding.dot(epEmbedding)
      val exp: Double = math.exp(-score)
      var objective: Double = 0.0
      var factor: Double = 0.0
      if (code(d) == 1) {
        objective = -math.log1p(exp)
        factor = exp / (1 + exp)
      }
      else if (code(d) == 0) {
        objective = -score - math.log1p(exp)
        factor = -1 / (1 + exp)
      }
      if (value ne null) value.accumulate(objective)
      if (gradient ne null) {
        //wordContexts.foreach(context => gradient.accumulate(model.weights(context), nodeEmbedding, factor))
        gradient.accumulate(model.weights(ep), nodeEmbedding, factor)
        gradient.accumulate(model.nodeWeights(currNode), epEmbedding, factor)
      }
    }
  }
}