package cc.factorie.app.nlp.embeddings.transRelations

import java.util.concurrent.atomic.AtomicInteger

import cc.factorie.app.nlp.embeddings._
import cc.factorie.la.{DenseTensor1, WeightsMapAccumulator}
import cc.factorie.model.Weights
import cc.factorie.optimize._
import cc.factorie.util.{DoubleSeq, DoubleAccumulator}

/**
 * Created by pat on 4/3/15.
 */
class TransH(opts: TransRelationOpts) extends TransRelationModel(opts) {

  var hyperPlanes: Seq[Weights] = null
  val epsilon = 0.1

  // Component-2
  def trainModel(trainTriplets: Seq[(String, String, String)]): Unit = {
    println("Learning Embeddings")
    val batchSize = trainTriplets.size / numBatches

    //    optimizer = new AdaGradRDA(delta = adaGradDelta, rate = adaGradRate)
    optimizer = new AdaGrad(delta = adaGradDelta, rate = adaGradRate)
    //    trainer = new HogwildTrainer(weightsSet = this.parameters, optimizer = optimizer, nThreads = threads, maxIterations = Int.MaxValue)
    trainer = new OnlineTrainer(weightsSet = this.parameters, optimizer = optimizer, maxIterations = Int.MaxValue, logEveryN = batchSize - 1)

    weights = (0 until entityCount + relationCount).map(i => Weights(TensorUtils.setToRandom1(new DenseTensor1(D, 0), rand))) // initialized using wordvec random
    hyperPlanes = (0 until relationCount).map(i => Weights(TensorUtils.setToRandom1(new DenseTensor1(D, 0), rand))) // initialized using wordvec random

    optimizer.initializeWeights(this.parameters)

    //    // normalize relation embeddings once
    //    (entityCount until weights.size).par.foreach(weights(_).value.twoNormalize())
    println(weights.size, entityCount, entityVocab.size(), relationCount, relationVocab.size())

    for (iteration <- 0 to iterations) {
      println(s"Training iteration: $iteration")


      normalize(weights, exactlyOne = false)
      normalize(hyperPlanes, exactlyOne = true)
      orthoganal()
      (0 until numBatches).foreach(batch => {
        // sample random triplets for miniBatch
        val miniBatch = Seq.fill(batchSize)(trainTriplets(rand.nextInt(trainTriplets.size)))
        processMiniBatch(miniBatch)
      })
    }
    println("Done learning embeddings. ")
    //store()
  }

  /*
  Enforce that relation distance vectors and hyperplanes are orthoganal
   */
  def orthoganal(): Unit = {
    (0 until relationCount).par.foreach(i => {
      val dr = weights(i).value
      val wr = hyperPlanes(i).value
      wr.twoNormalize()
      val dot = dr.dot(wr)
      if (dot > epsilon){
        dr -= wr * adaGradRate
        wr -= dr * adaGradRate        
      }      
    })
    normalize(hyperPlanes, exactlyOne = true)
  }


  protected def processMiniBatch(relationList: Seq[(String, String, String)]): Unit = {
    val examples = relationList.map { case (e1, relation, e2) =>
      val e1Index = entityVocab.get(e1)
      val e2Index = entityVocab.get(e2)
      val relationIndex = relationVocab.get(relation) + entityCount
      // corrupt either head or tail
      if (rand.nextInt(2) == 0) new TansHExample(this, e1Index, relationIndex, e2Index, rand.nextInt(entityCount), e2Index, l1)
      else new TansHExample(this, e1Index, relationIndex, e2Index, e1Index, rand.nextInt(entityCount), l1)
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
    val i = new AtomicInteger(0)
    val tot = testTriplets.size.toDouble
    val ranks: Seq[Int] = testTriplets.par.flatMap { case (e1, relation, e2) =>
      val e1Id = entityVocab.get(e1)
      val e2Id = entityVocab.get(e2)
      val e1Emb = weights(e1Id).value
      val e2Emb = weights(e2Id).value
      val relId = relationVocab.get(relation)
      val relEmb = weights(relId + entityCount).value
      val hyperPlane = hyperPlanes(relId).value

      val e1Proj = e1Emb - hyperPlane.*(e1Emb.dot(hyperPlane))
      val e2Proj = e2Emb - hyperPlane.*(e2Emb.dot(hyperPlane))

      val posScore = (e1Proj + relEmb - e2Proj).oneNorm
      // store for efficiency
      val e1Rel = e2Proj + relEmb
      val relE2 = relEmb - e2Proj

      var headRank = 0
      var tailRank = 0
      // iterate over each other entity in dictionary
      var negativeId = 0
      while (negativeId < entityCount) {
        // dont self rank
        if (negativeId != e1Id || negativeId != e2Id) {
          val negEmb = weights(negativeId).value
          val negProj = negEmb - hyperPlane.*(negEmb.dot(hyperPlane))

          if (negativeId != e1Id) {
            val negHeadScore = (e1Rel - negProj).oneNorm
            if (negHeadScore < posScore)
              headRank += 1
          }
          if (negativeId != e2Id) {
            val negTailScore = (negProj - relE2).oneNorm
            if (negTailScore < posScore)
              tailRank += 1
          }
        }
        negativeId += 1
      }
      val tmp = i.incrementAndGet()
      if (tmp % 1000 == 0) println(tmp/tot)
      Seq(headRank, tailRank)
    }.seq
    // return hits@10 and avg rank
    (ranks.count(_ < 10).toDouble / ranks.size.toDouble, ranks.sum / ranks.length)
  }

}

class TansHExample(model: TransH, e1PosDex: Int, relDex: Int, e2PosDex: Int, e1NegDex: Int, e2NegDex: Int, l1: Boolean = false) extends Example {

  val factor: Double = 1.0

  def accumulateValueAndGradient(value: DoubleAccumulator, gradient: WeightsMapAccumulator): Unit = {
    val e1PosEmb = model.weights(e1PosDex).value
    val e1NegEmb = model.weights(e1NegDex).value

    val e2PosEmb = model.weights(e2PosDex).value
    val e2NegEmb = model.weights(e2NegDex).value

    val relEmb = model.weights(relDex).value
    // gross indexing
    val hPlaneDex = relDex - model.entityCount
    val hyperPlane = model.hyperPlanes(hPlaneDex).value

    val posGrad = e2PosEmb - hyperPlane.*(e1PosEmb.dot(hyperPlane)) - relEmb
    val negGrad = e2NegEmb - hyperPlane.*(e1NegEmb.dot(hyperPlane)) - relEmb

    //    val constraints =
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
      gradient.accumulate(model.hyperPlanes(hPlaneDex), posGrad, factor)
      gradient.accumulate(model.weights(e1NegDex), negGrad, -factor)
      gradient.accumulate(model.weights(e2NegDex), negGrad, factor)
      gradient.accumulate(model.weights(relDex), negGrad, -factor)
      gradient.accumulate(model.hyperPlanes(hPlaneDex), negGrad, -factor)
    }
  }
}

object TestTransH extends App {

  val opts = new TransRelationOpts()
  opts.parse(args)

  val transH = new TransH(opts)
  val train = transH.buildVocab(opts.train.value, transH.parseTsv)
  val test = transH.buildVocab(opts.test.value, transH.parseTsv)
  println(train.size, test.size)
  transH.trainModel(train)
    println(transH.evaluate(test))

}
