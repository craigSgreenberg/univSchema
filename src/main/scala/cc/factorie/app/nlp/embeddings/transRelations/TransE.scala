package cc.factorie.app.nlp.embeddings.transRelations

import cc.factorie.app.nlp.embeddings.TensorUtils
import cc.factorie.la.{DenseTensor1, WeightsMapAccumulator}
import cc.factorie.optimize._
import cc.factorie.util.DoubleAccumulator

/**
 * Created by pat on 4/3/15.
 */
class TransE(opts : TransRelationOpts) extends TransRelationModel(opts)
{

  // Component-2
  def trainModel(trainTriplets: Seq[(String, String, String)]): Unit = {
    println("Learning Embeddings")
    val batchSize = trainTriplets.size / numBatches

    //    optimizer = new AdaGradRDA(delta = adaGradDelta, rate = adaGradRate)
    optimizer = new AdaGrad(delta = adaGradDelta, rate = adaGradRate)
        trainer = new HogwildTrainer(weightsSet = this.parameters, optimizer = optimizer, nThreads = threads, maxIterations = Int.MaxValue, logEveryN = batchSize-1)
//    trainer = new OnlineTrainer(weightsSet = this.parameters, optimizer = optimizer, maxIterations = Int.MaxValue, logEveryN = batchSize-1)

    weights = (0 until entityCount + relationCount).map(i => Weights(TensorUtils.setToRandom1(new DenseTensor1(D, 0), rand))) // initialized using wordvec random
    optimizer.initializeWeights(this.parameters)

    println(weights.size, entityCount, entityVocab.size(), relationCount, relationVocab.size())

    for (iteration <- 0 to iterations) {
      println(s"Training iteration: $iteration")

      normalize(weights, exactlyOne = true)

      (0 until numBatches).foreach(batch => {
        // sample random triplets for miniBatch
        val miniBatch = Seq.fill(batchSize)(trainTriplets(rand.nextInt(trainTriplets.size)))
        processMiniBatch(miniBatch)
      })
    }
    println("Done learning embeddings. ")
    //store()
  }

  protected def processMiniBatch(relationList: Seq[(String, String, String)]): Unit = {
    val examples = relationList.map { case (e1, relation, e2) =>
      val e1Index = entityVocab.get(e1)
      val e2Index = entityVocab.get(e2)
      val relationIndex = relationVocab.get(relation) + entityCount
      // corrupt either head or tail
      if (rand.nextInt(2) == 0) new TansEExample(this, e1Index, relationIndex, e2Index, rand.nextInt(entityCount), e2Index, l1)
      else new TansEExample(this, e1Index, relationIndex, e2Index, e1Index, rand.nextInt(entityCount), l1)
    }
    trainer.processExamples(examples)
  }

  /**
   * for each test triplet, rank the correct answer amongst all corrupted head triplets
   * and all corrupted tail triplets
   * @param testTriplets test triples in form e1 relation e2
   * @return (hits@10, averageRank)
   */
  def evaluate(testTriplets: Seq[(String, String, String)]): (Double, Double) = {

    println(s"Evaluating on ${testTriplets.size} samples")
    val ranks: Seq[Int] = testTriplets.par.flatMap { case (e1, relation, e2) =>
      val e1Id = entityVocab.get(e1)
      val e2Id = entityVocab.get(e2)
      val e1Emb = weights(e1Id).value
      val e2Emb = weights(e2Id).value
      val relEmb = weights(relationVocab.get(relation) + entityCount).value

      var headRank = 0
      var tailRank = 0
      val posScore = if (l1) (e1Emb + relEmb - e2Emb).oneNorm else (e1Emb + relEmb - e2Emb).twoNorm

      // store to save time
      val e1Rel = e1Emb + relEmb
      val relE2 = relEmb - e2Emb

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
      Seq(headRank, tailRank)
    }.seq
    // return hits@10 and avg rank
    (ranks.count(_ < 10).toDouble / ranks.size.toDouble, ranks.sum / ranks.length)
  }
}

class TansEExample(model: TransRelationModel, e1PosDex: Int, relDex: Int, e2PosDex: Int, e1NegDex: Int, e2NegDex: Int, l1: Boolean = false) extends Example {

  val factor: Double = 1.0

  def accumulateValueAndGradient(value: DoubleAccumulator, gradient: WeightsMapAccumulator): Unit = {
    val e1PosEmb = model.weights(e1PosDex).value
    val e1NegEmb = model.weights(e1NegDex).value

    val e2PosEmb = model.weights(e2PosDex).value
    val e2NegEmb = model.weights(e2NegDex).value

    val relEmb = model.weights(relDex).value

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
      gradient.accumulate(model.weights(relDex), posGrad, factor)
      gradient.accumulate(model.weights(e1NegDex), negGrad, -factor)
      gradient.accumulate(model.weights(e2NegDex), negGrad, factor)
      gradient.accumulate(model.weights(relDex), negGrad, -factor)

    }
  }
}

object TestTransE extends App {

  val opts = new TransRelationOpts()
  opts.parse(args)

  val transE = new TransE(opts)
  // if data is in format [r1,r2 rel score] use parseArvind
  val train = transE.buildVocab(opts.train.value, transE.parseTsv) //transE.parseArvind)
  val test = transE.buildVocab(opts.test.value, transE.parseTsv) //transE.parseArvind)
  println(train.size, test.size)
  transE.trainModel(train)
  println(transE.evaluate(test))

}
