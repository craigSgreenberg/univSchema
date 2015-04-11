package cc.factorie.app.nlp.embeddings

import scala.collection.mutable
import cc.factorie.optimize.Example
import cc.factorie.util.DoubleAccumulator
import cc.factorie.la.{DenseTensor1, WeightsMapAccumulator}


class NegativeSampling (override val opts: EmbeddingOpts) extends UniversalSchemaModel(opts) {
  override def process(ep: Int, rel: Int): Unit = {
        trainer.processExample(new NegativeSamplingExample(this, ep, rel))
  }
  override def getScore(ep: Int, rel: Int): Double = {
    val nodeEmbedding = nodeWeights(rel).value
    val epEmbedding =   weights(ep).value
    val ans: Double = nodeEmbedding.dot(epEmbedding)
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



class NegativeSamplingExample(model: UniversalSchemaModel, ep: Int, rel: Int) extends Example {
  val attempts = 20
  def getNegEp(): Int = {
    var trial = 0
    var found = false
    var ret = -1
    while(trial < attempts && (!found)){
      val neg = model.rand.nextInt(model.entPairSize)
      if(!(model.positives(rel).contains(neg))){
        found = true
        ret = neg
      }
      trial += 1
    }
    ret
  }

  def accumulateValueAndGradient(value: DoubleAccumulator, gradient: WeightsMapAccumulator): Unit = {
    /*
    val contextEmbedding = new DenseTensor1(model.D, 0)
    wordContexts.foreach(context => contextEmbedding.+=(model.weights(context).value))
    contextEmbedding.+=(model.parWeights(parContext).value)
    */
      //println(currNode)
      for(j <- 0 until model.opts.negative.value) {
        var negEp = getNegEp()
        if(negEp == -1) return

        model.processed += 1
        val nodeEmbedding = model.nodeWeights(rel).value
        val epEmbedding =   model.weights(ep).value
        val negEpEmbedding =   model.weights(negEp).value
        //if(model.opts)
        if(model.opts.hinge.value){
          var diff: Double = model.getScore(ep, rel) - model.getScore(negEp, rel) - model.opts.margin.value
          if(model.opts.wsabie.value){
            val attempts = 10
            var trial = 0
            var found = false
            val posScore = model.getScore(ep, rel)
            while(trial < attempts && (!found)){
              if(diff < 0)  found = true
              else{
                negEp = getNegEp()
                diff = posScore - model.getScore(negEp, rel) - model.opts.margin.value
                trial += 1
              }
            }
          }
          val objective: Double = 0.0
          val factor: Double = diff > 0 match {
            case true =>  0.0
            case false => 1.0
          }
          if (value ne null) value.accumulate(objective)
          if (gradient ne null) {
            //wordContexts.foreach(context => gradient.accumulate(model.weights(context), nodeEmbedding, factor))
            gradient.accumulate(model.weights(ep), nodeEmbedding, factor)
            gradient.accumulate(model.weights(negEp), nodeEmbedding, -1 * factor)
            gradient.accumulate(model.nodeWeights(rel), (epEmbedding - negEpEmbedding), factor)
          }
        }
        else{
          val diff: Double = model.getScore(ep, rel) - model.getScore(negEp, rel)
          val objective: Double = 0.0
          var factor: Double = 1/(1 + math.exp(-diff))
          factor = 1 - factor
          if (value ne null) value.accumulate(objective)
          if (gradient ne null) {
            //wordContexts.foreach(context => gradient.accumulate(model.weights(context), nodeEmbedding, factor))
            gradient.accumulate(model.weights(ep), nodeEmbedding, factor)
            gradient.accumulate(model.weights(negEp), nodeEmbedding, -1 * factor)
            gradient.accumulate(model.nodeWeights(rel), (epEmbedding - negEpEmbedding), factor)
          }
        }
      }
    }
}