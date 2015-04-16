package cc.factorie.app.nlp.embeddings.transRelations

import java.util.concurrent.atomic.AtomicInteger

import cc.factorie.app.nlp.embeddings.{EmbeddingOpts, LiteHogwildTrainer, TensorUtils}
import cc.factorie.la.{DenseTensor1, WeightsMapAccumulator}
import cc.factorie.model.Weights
import cc.factorie.optimize._
import cc.factorie.util.{Threading, DoubleAccumulator}

import scala.collection.mutable.ArrayBuffer

/**
 * Created by pat on 4/3/15.
 */
class TransE(opts: EmbeddingOpts) extends TransRelationModel(opts) {

  // Component-2
  override def learnEmbeddings(): Unit = {
    println("Learning Embeddings")
//    optimizer = new ConstantLearningRate(adaGradRate)
    optimizer = new AdaGradRDA(delta = adaGradDelta, rate = adaGradRate, l2=opts.regularizer.value)
    trainer = new LiteHogwildTrainer(weightsSet = this.parameters, optimizer = optimizer, nThreads = threads, maxIterations = Int.MaxValue)
    //    trainer = new OnlineTrainer(weightsSet = this.parameters, optimizer = optimizer, maxIterations = Int.MaxValue, logEveryN = batchSize-1)

    weights = (0 until entityCount + relationSize).map(i => Weights(TensorUtils.setToRandom1(new DenseTensor1(D, 0), rand))) // initialized using wordvec random
    optimizer.initializeWeights(this.parameters)

    for (iteration <- 1 to iterations) {
      println(s"Training iteration: $iteration")
      val st1 = System.currentTimeMillis()
//      normalize(weights, exactlyOne = true)
      val batches = (0 until (trainingExamples.size/batchSize)).map(batch => new MiniBatchExample(generateMiniBatch()))
//      val batches = trainingExamples.map(e => makeExample(e._2, e._4, e._3))
      val st = System.currentTimeMillis()
      println("comuting gradients " + (st - st1) / 1000.0)
      trainer.processExamples(batches)
      val st2 = System.currentTimeMillis()
      println("finished comuting gradients " + (st2 - st) / 1000.0)
      if(iteration % opts.evalautionFrequency.value == 0) {
        println("Dev MAP after " + iteration + " iterations: " + evaluate(opts.devFile.value, iteration))
        println("Test MAP after " + iteration + " iterations: " + evaluate(opts.testFile.value, iteration))
      }
    }
    println("Done learning embeddings. ")
    //store()
  }

  def makeExample(e1Index : Int, relationIndex : Int, e2Index : Int): Example ={
    new TransEExample(this, e1Index, relationIndex, e2Index, l1)
  }


  override def getScore(ep: String, rel: String): Double = {
    val Array(e1, e2) = ep.split(",")
    getScore((e1, rel, e2))
  }
  
  /**
   * Score a relation triplet
   * @param triple (e1, rel, e2)
   * @return
   */
  def getScore(triple: (String, String, String)): Double = {
    val (e1, rel, e2) = triple
    assert(entityVocab.containsKey(e1) && entityVocab.containsKey(e2) && relationKey.containsKey(rel),
      "Something was not in the vocab. Sorry")
    val e1Emb = weights(entityVocab.get(e1)).value
    val e2Emb = weights(entityVocab.get(e2)).value
    val relEmb = weights(relationKey.get(rel) + entityCount).value
    getScore(e1Emb, e2Emb, relEmb)
  }

  def getScore (e1Emb: Weights#Value, e2Emb: Weights#Value, relEmb: Weights#Value): Double =
  {
    val result = e1Emb + relEmb - e2Emb
    if (l1) result.oneNorm else result.twoNorm
  }

  /**
   * for each test triplet, rank the correct answer amongst all corrupted head triplets
   * and all corrupted tail triplets
   * @param testTriplets test triples in form e1 relation e2
   * @return (hits@10, averageRank)
   */
  def avgRankHitsAt10(testTriplets: Seq[(String, String, String)]): (Double, Double) = {

    println(s"Evaluating on ${testTriplets.size} samples")
    val i = new AtomicInteger(0)
    val tot = testTriplets.size.toDouble
    val ranks: Seq[Int] = testTriplets.par.flatMap { case (e1, relation, e2) =>
      val e1Id = entityVocab.get(e1)
      val e2Id = entityVocab.get(e2)
      val e1Emb = weights(e1Id).value
      val e2Emb = weights(e2Id).value
      val relEmb = weights(relationKey.get(relation) + entityCount).value

      var headRank = 0
      var tailRank = 0
      // store to save time
      val e1Rel = e1Emb + relEmb
      val relE2 = relEmb - e2Emb

      val posScore = if (l1) (e1Rel - e2Emb).oneNorm else (e1Emb + relE2).twoNorm

      // iterate over each other entity in dictionary
      var negativeId = 0
      while (negativeId < entityCount) {
        // dont self rank
        if (negativeId != e1Id && negativeId != e2Id) {
          val negEmb = weights(negativeId).value
          val negHeadScore = if (l1) (e1Rel - negEmb).oneNorm else (e1Rel - negEmb).twoNorm
          if (negHeadScore < posScore)
            headRank += 1
          val negTailScore = if (l1) (negEmb + relE2).oneNorm else (negEmb + relE2).twoNorm
          if (negTailScore < posScore)
            tailRank += 1
        }
        negativeId += 1
      }
      val tmp = i.incrementAndGet()
      if (tmp % 1000 == 0) println(tmp / tot)
      Seq(headRank, tailRank)
    }.seq
    // return hits@10 and avg rank
    (ranks.count(_ < 10).toDouble / ranks.size.toDouble, ranks.sum / ranks.length)
  }

  // override this function in your Embedding Model like SkipGramEmbedding or CBOWEmbedding
  override protected def process(ep: Int, rel: Int): Unit = ???

  override def getScore(ep: Int, rel: Int): Double = ???
}

class TransEExample(model: TransRelationModel, e1PosDex: Int, relDex: Int, e2PosDex: Int, l1: Boolean = false) extends Example {

  val factor: Double = 1.0

  def accumulateValueAndGradient(value: DoubleAccumulator, gradient: WeightsMapAccumulator): Unit =
  {
//    println("examplin")
    val e1PosEmb = model.weights(e1PosDex).value
    val e2PosEmb = model.weights(e2PosDex).value
    val relEmb = model.weights(relDex).value

    var negSample = 0
    while (negSample < model.negativeSamples) {
      // draw negative sample randomly either from head or tail
      val (e1NegDex, e2NegDex) = model.negativeSample(e1PosDex, e2PosDex, relDex-model.entityCount, None)

      val e1NegEmb = model.weights(e1NegDex).value
      val e2NegEmb = model.weights(e2NegDex).value

      val posGrad = e2PosEmb - e1PosEmb - relEmb
      val negGrad = e2NegEmb - e1NegEmb - relEmb

      // gamma + pos - neg
      val objective = if (l1) model.gamma + posGrad.oneNorm - negGrad.oneNorm
      else model.gamma + posGrad.twoNorm - negGrad.twoNorm

      if (l1) {
        (0 until posGrad.size).foreach(i => {
          posGrad(i) = if (posGrad(i) > 0) 1.0 else -1.0
          negGrad(i) = if (negGrad(i) > 0) 1.0 else -1.0
        })
      }

      if (value ne null) value.accumulate(objective)

      // hinge loss
      if (gradient != null && objective > 0.0) {
        gradient.accumulate(model.weights(e1PosDex), posGrad, factor)
        gradient.accumulate(model.weights(e2PosDex), posGrad, -factor)
        gradient.accumulate(model.weights(relDex), posGrad - negGrad, factor)
        gradient.accumulate(model.weights(e1NegDex), negGrad, -factor)
        gradient.accumulate(model.weights(e2NegDex), negGrad, factor)
//        gradient.accumulate(model.weights(relDex), negGrad, -factor)

      }
      negSample += 1
    }
  }
}

object TestTransE extends App {

  val opts = new EmbeddingOpts()
  opts.parse(args)

  val transE = new TransE(opts)
  // if data is in format [r1,r2 rel score] use parseArvind
  val train = transE.buildVocab()
  val test = transE.fileToTriplets(opts.testFile.value).toSeq.flatMap(eList => eList._2.toSet.toSeq)
//  println(train.size, test.size)
  transE.learnEmbeddings()
  println(transE.avgRankHitsAt10(test))

}
