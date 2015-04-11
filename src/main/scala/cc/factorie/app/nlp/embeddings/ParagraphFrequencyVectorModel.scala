package cc.factorie.app.nlp.embeddings
import scala.collection.mutable.HashMap
import scala.collection.mutable
import cc.factorie.optimize.Example
import cc.factorie.util.DoubleAccumulator
import cc.factorie.la.{DenseTensor1, WeightsMapAccumulator}
/*
class ParagraphFrequencyVectorModel (override val opts: EmbeddingOpts) extends WordEmbeddingModel(opts) {

  val sample = opts.sample.value.toDouble
  val windowLength = 10
  val rng = new util.Random()

  override def process(doc: String): Int = {

    if (doc.size == 0) {
      println("doc size 0")
      return 0
    }

    val fullsen = doc.stripLineEnd.split(' ')



    val docId = fullsen(0).toInt

    var sen = fullsen.drop(1).map(word => vocab.getId(word)).filter(id => id != -1)

    val wordCount = sen.size

    // storing the actual distribution of words for a document
    var docWordFreq=new HashMap[Int,Int]().withDefaultValue(0)
    var nodeFreq = new HashMap[Int,Int]().withDefaultValue(0)

    sen.foreach{w =>
       docWordFreq(w)+=1

    }



    docWordFreq foreach{ pair=>
      var w = pair._1
      var freq = pair._2
      var codeLen =  vocab.getCodelength(w)
      var point = vocab.getPoint(w)

      for (d <- 0 until codeLen) {
        val currNode = point(d)
        nodeFreq(currNode) +=  freq

      }
    }



    val nodeDist = nodeFreq.mapValues(m=> (m.toDouble / wordCount.toDouble))

    /*if (sample > 0)
      sen = sen.filter(id => subSample(id) != -1) */

    val senLength = sen.size
    if(senLength >= windowLength){
      for (senPosition <- 0 until (senLength / windowLength).toInt) {
        val b = rng.nextInt(senLength)
        val wordContexts = new mutable.ArrayBuffer[Int]

        val end = windowLength + b


        if(end < senLength){
          for (a <- b until end) {
            wordContexts += sen(a)
          }

          val windowLen = wordContexts.size
          var contextWordIndex = rng.nextInt(windowLen)
          var contextWord =  wordContexts(contextWordIndex)
          var currWord = wordContexts(rng.nextInt(windowLen))
          if(contextWord==currWord)
          {
            if(contextWordIndex>=windowLen-1){
              contextWord =  wordContexts(windowLen-2)
              currWord = wordContexts(windowLen-1)
            }
            else
            {
               contextWordIndex=contextWordIndex+1
               currWord= wordContexts(contextWordIndex)
            }
          }
          /*while(contextWord==currWord){
             currWord = wordContexts(rng.nextInt(windowLen))
          }*/

          val parContext = docId
          trainer.processExample(new ParagraphVectorFrequencyHierarchicalSoftMaxExample(this, currWord, contextWord,parContext,vocab.getCodelength(currWord),vocab.getCode(currWord),vocab.getPoint(currWord),nodeDist))
        }

      }
    }
    return wordCount
  }

}

class ParagraphVectorFrequencyHierarchicalSoftMaxExample(model: WordEmbeddingModel, word: Int, contextWord: Int,parContext: Int, codeLen:Int,code: Array[Int],point:Array[Int],nodeDist:scala.collection.Map[Int,Double]) extends Example {
  def accumulateValueAndGradient(value: DoubleAccumulator, gradient: WeightsMapAccumulator): Unit = {
    val contextEmbedding = new DenseTensor1(model.D, 0)
    contextEmbedding.+=(model.weights(contextWord).value)
    contextEmbedding.+=(model.parWeights(parContext).value)
    //println("##########################")
    for (d <- 0 until codeLen) {

      val currNode = point(d)

      val currNodeDist = nodeDist(currNode)
      val nodeEmbedding = model.nodeWeights(currNode).value
      val score: Double = nodeEmbedding.dot(contextEmbedding)
      val exp: Double = math.exp(-score)

      var objective: Double = 0.0
      var factor: Double = 0.0

      //cross entropy
      if (code(d) == 1) {
        objective =  - (math.log1p(exp) * currNodeDist)
        factor =   (exp * currNodeDist) / (1 + exp)
      }
      if (code(d) == 0) {
        objective = - (score + math.log1p(exp)) * currNodeDist
        factor = - currNodeDist / (1 + exp)
      }

     /* if (code(d) == 1) {
        objective =  - (math.log1p(exp) )
        factor =   (exp ) / (1 + exp)
        //println("1 Objective "+objective)
        //println("1 Actual "+currNodeDist+"\t"+"Predicted "+ 1 / (1+exp))
      }
      if (code(d) == 0) {
        objective =  - (score + math.log1p(exp))
        factor = - 1 / (1 + exp)
        //println("0 Objective "+objective)
        //println("0 Actual "+currNodeDist+"\t"+"Predicted "+ exp / (1+exp))
      } */



      /*if (code(d) == 1) {
        objective =  - ((1 / (1 + exp)) -  currNodeDist) * ((1 / (1 + exp)) -  currNodeDist)
        factor =    2 * ((1 / (1 + exp)) -  currNodeDist) * exp
      }
      if (code(d) == 0) {
        objective = - (score + math.log1p(exp)) * currNodeDist
        factor = - currNodeDist / (1 + exp)
      }  */

      if (value ne null) value.accumulate(objective)
      if (gradient ne null) {
        gradient.accumulate(model.weights(contextWord), nodeEmbedding, factor)
        gradient.accumulate(model.parWeights(parContext), nodeEmbedding,factor)
        gradient.accumulate(model.nodeWeights(currNode), contextEmbedding, factor)
      }

    }
  }
}
*/