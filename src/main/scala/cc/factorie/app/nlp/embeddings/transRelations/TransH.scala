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
class TransH(opts: EmbeddingOpts) extends TransRelationModel(opts) {

  var hyperPlanes: Seq[Weights] = null
  val epsilon = 0.1
  val epsilonSquared = epsilon*epsilon
  val C = 0.015625
  var softConstraints = 0.0

  // Component-2
  def trainModel(trainTriplets: Seq[(String, String, String)]): Unit = {
    println("Learning Embeddings")
//        optimizer = new ConstantLearningRate(adaGradRate)
    optimizer = new AdaGradRDA(delta = adaGradDelta, rate = adaGradRate)
    trainer = new LiteHogwildTrainer(weightsSet = this.parameters, optimizer = optimizer, nThreads = threads, maxIterations = Int.MaxValue)
//    trainer = new OnlineTrainer(weightsSet = this.parameters, optimizer = optimizer, maxIterations = Int.MaxValue, logEveryN = batchSize-1)

    weights = (0 until entityCount + relationSize).map(i => Weights(TensorUtils.setToRandom1(new DenseTensor1(D, 0), rand))) // initialized using wordvec random
    hyperPlanes = (0 until relationSize).map(i => Weights(TensorUtils.setToRandom1(new DenseTensor1(D, 0), rand))) // initialized using wordvec random

    optimizer.initializeWeights(this.parameters)

    for (iteration <- 0 to iterations) {
      println(s"Training iteration: $iteration")

      normalize(weights, exactlyOne = false)
      normalize(hyperPlanes, exactlyOne = true)
      (0 until relationSize).foreach(i => orthoganal(weights(i+entityCount).value, hyperPlanes(i).value))
      
      softConstraints = calculateSoftConstraints()
      val batches = (0 until (trainingExamples.size/batchSize)).map(batch => new MiniBatchExample(generateMiniBatch()))
      trainer.processExamples(batches)
    }
    println("Done learning embeddings. ")
    //store()
  }

  def makeExample(e1Index: Int, relationIndex: Int, e2Index: Int): Example = {
    new TransHExample(this, e1Index, relationIndex, e2Index, l1)
  }

  /*
  Enforce that relation distance vectors and hyperplanes are orthoganal
   */
  def orthoganal(embedding : Weights#Value, hPlane : Weights#Value): Unit = {
    hPlane.twoNormalize()
    val dot = embedding.dot(hPlane)
    if (dot > epsilon) {
      embedding -= hPlane * adaGradRate
      hPlane -= embedding * adaGradRate
      hPlane.twoNormalize()
    }
  }

  def calculateSoftConstraints(): Double =
  {
    val entityScore = weights.slice(0, entityCount).map(e => Math.max(0, e.value.twoNormSquared - 1.0)).sum
    val relationProjectionScore = (0 until relationSize).map(i => {
      val dr = weights(i + entityCount).value
      val wr = hyperPlanes(i).value
      val dot = dr.dot(wr)
      Math.max(0, (dot*dot / dr.twoNormSquared) / epsilonSquared)
    }).sum
    C * (entityScore + relationProjectionScore)
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
    val relDex = relationKey.get(rel)
    // gross indexing
    val relEmb = weights(relDex + entityCount).value
    val hPlaneDex = relDex
    val hyperPlane = hyperPlanes(hPlaneDex).value

    getScore(e1Emb, e2Emb, relEmb, hyperPlane)

  }

  def getScore (e1Emb: Weights#Value, e2Emb: Weights#Value, relEmb: Weights#Value, hyperPlane: Weights#Value): Double =
  {
    val result = e2Emb - hyperPlane.*(e1Emb.dot(hyperPlane)) - relEmb
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
      val relId = relationKey.get(relation)
      val relEmb = weights(relId + entityCount).value
      val hyperPlane = hyperPlanes(relId).value

      val e1Proj = e1Emb - hyperPlane.*(e1Emb.dot(hyperPlane))
      val e2Proj = e2Emb - hyperPlane.*(e2Emb.dot(hyperPlane))

      val posScore = if (l1) (e1Proj + relEmb - e2Proj).oneNorm else (e1Proj + relEmb - e2Proj).twoNorm
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
            val negHeadScore = if (l1) (e1Rel - negProj).oneNorm else (e1Rel - negProj).twoNorm
            println(posScore, negHeadScore)
            if (negHeadScore < posScore)
              headRank += 1
          }
          if (negativeId != e2Id) {
            val negTailScore = if (l1) (negProj - relE2).oneNorm else (negProj - relE2).twoNorm
            println(posScore, negTailScore)

            if (negTailScore < posScore)
              tailRank += 1
          }
        }
        negativeId += 1
      }
      val tmp = i.incrementAndGet()
      if (tmp % 1000 == 0) println(tmp / tot)
      println(headRank, tailRank)
      Seq(headRank, tailRank)
    }.seq
    // return hits@10 and avg rank
    (ranks.count(_ < 10).toDouble / ranks.size.toDouble, ranks.sum / ranks.length)
  }

  // override this function in your Embedding Model like SkipGramEmbedding or CBOWEmbedding
  override protected def process(ep: Int, rel: Int): Unit = ???

  override def getScore(ep: Int, rel: Int): Double = ???
}

class TransHExample(model: TransH, e1PosDex: Int, relDex: Int, e2PosDex: Int, l1: Boolean = false) extends Example {

  val factor: Double = 1.0

  def accumulateValueAndGradient(value: DoubleAccumulator, gradient: WeightsMapAccumulator): Unit = {
    val e1PosEmb = model.weights(e1PosDex).value
    val e2PosEmb = model.weights(e2PosDex).value
    val relEmb = model.weights(relDex).value
    // gross indexing
    val hPlaneDex = relDex - model.entityCount
    val hyperPlane = model.hyperPlanes(hPlaneDex).value

    var negSample = 0
    while (negSample < model.negativeSamples) {
      // draw negative sample randomly either from head or tail
      val (e1NegDex, e2NegDex) =
        if (model.rand.nextInt(2) == 0) (e1PosDex, model.negativeSampleEntity(None))
        else (model.negativeSampleEntity(None), e2PosDex)

      val e1NegEmb = model.weights(e1NegDex).value
      val e2NegEmb = model.weights(e2NegDex).value

      val posGrad = e2PosEmb - hyperPlane.*(e1PosEmb.dot(hyperPlane)) - relEmb
      val negGrad = e2NegEmb - hyperPlane.*(e1NegEmb.dot(hyperPlane)) - relEmb

      // gamma + pos - neg
      val objective = (if (l1) model.gamma + posGrad.oneNorm - negGrad.oneNorm
      else model.gamma + posGrad.twoNorm - negGrad.twoNorm) + model.softConstraints

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
      negSample += 1
    }
  }
}

object TestTransH extends App
{
  val opts = new EmbeddingOpts()
  opts.parse(args)

  val transH = new TransH(opts)
  val train = transH.buildVocab()
  val test = transH.fileToTriplets(opts.testFile.value).toSeq.flatMap(eList => eList._2.toSet.toSeq)
  transH.learnEmbeddings()
  println(Evaluator.avgRankHitsAt10(transH, test))

}
