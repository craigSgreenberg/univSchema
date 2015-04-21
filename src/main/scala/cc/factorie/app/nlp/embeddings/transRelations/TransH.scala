package cc.factorie.app.nlp.embeddings.transRelations

import java.util.concurrent.atomic.AtomicInteger

import cc.factorie.app.nlp.embeddings.{Evaluator, EmbeddingOpts, LiteHogwildTrainer, TensorUtils}
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
  override def learnEmbeddings(): Unit = {
    println("Learning Embeddings")
        optimizer = new ConstantLearningRate(adaGradRate)
//    optimizer = new AdaGradRDA(delta = adaGradDelta, rate = adaGradRate, r)
    trainer = new LiteHogwildTrainer(weightsSet = this.parameters, optimizer = optimizer, nThreads = threads, maxIterations = Int.MaxValue)
//    trainer = new OnlineTrainer(weightsSet = this.parameters, optimizer = optimizer, maxIterations = Int.MaxValue, logEveryN = batchSize-1)

    weights = (0 until entityCount + relationSize).map(i => Weights(TensorUtils.setToRandom1(new DenseTensor1(D, 0), rand))) // initialized using wordvec random
    hyperPlanes = (0 until relationSize).map(i => Weights(TensorUtils.setToRandom1(new DenseTensor1(D, 0), rand))) // initialized using wordvec random

    optimizer.initializeWeights(this.parameters)
    val nBatches = trainingExamplesSize/batchSize

    for (iteration <- 0 until iterations) {
      println(s"Training iteration: $iteration")
      val st1 = System.currentTimeMillis()

      normalize(weights, exactlyOne = false)
      normalize(hyperPlanes, exactlyOne = true)
      (0 until relationSize).foreach(i => orthoganal(weights(i+entityCount).value, hyperPlanes(i).value))
      
      softConstraints = calculateSoftConstraints()
      val batches = (0 until nBatches).map(batch => new MiniBatchExample(generateMiniBatch()))
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
      Math.max(0, (dot*dot / dr.twoNormSquared) - epsilonSquared)
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

    val e1Proj = e1Emb - hyperPlane.*(e1Emb.dot(hyperPlane))
    val e2Proj = e2Emb - hyperPlane.*(e2Emb.dot(hyperPlane))

    val result = e1Proj + relEmb - e2Proj
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

      // store for efficiency
      val e1Rel = e1Proj + relEmb
      val relE2 = relEmb - e2Proj

      val posScore = if (l1) (e1Proj + relEmb - e2Proj).oneNorm else (e1Proj + relEmb - e2Proj).twoNorm

      var headRank = 0
      var tailRank = 0
      // iterate over each other entity in dictionary
      var negativeId = 0
      while (negativeId < entityCount) {
        // dont self rank
        if (negativeId != e1Id && negativeId != e2Id) {
          val negEmb = weights(negativeId).value
          val negProj = negEmb - hyperPlane.*(negEmb.dot(hyperPlane))

          if (negativeId != e1Id) {
            val negHeadScore = if (l1) (e1Rel - negProj).oneNorm else (e1Rel - negProj).twoNorm
            if (negHeadScore < posScore)
              headRank += 1
          }
          if (negativeId != e2Id) {
            val negTailScore = if (l1) (negProj - relE2).oneNorm else (negProj - relE2).twoNorm

            if (negTailScore < posScore)
              tailRank += 1
          }
        }
        negativeId += 1
      }
      val tmp = i.incrementAndGet()
      if (tmp % 1000 == 0) println(tmp / tot)
//      println(headRank, tailRank)
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
    // project e1 and e2
    val e1PosProj = e1PosEmb - hyperPlane.*(e1PosEmb.dot(hyperPlane))
    val e2PosProj = e2PosEmb - hyperPlane.*(e2PosEmb.dot(hyperPlane))

    var negSample = 0
    while (negSample < model.negativeSamples) {
      // draw negative sample randomly either from head or tail
      val (e1NegDex, e2NegDex) = model.negativeSample(e1PosDex, e2PosDex, relDex-model.entityCount, None)

      val e1NegEmb = model.weights(e1NegDex).value
      val e2NegEmb = model.weights(e2NegDex).value
      val e1NegProj = e1NegEmb - hyperPlane.*(e1NegEmb.dot(hyperPlane))
      val e2NegProj = e2NegEmb - hyperPlane.*(e2NegEmb.dot(hyperPlane))

      // gradients
      val posGrad = e2PosProj - e1PosProj - relEmb
      val negGrad = e2NegProj - e1NegProj - relEmb
      posGrad.twoNormalize()
      negGrad.twoNormalize()

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
        // scale by (1-w)
        val oneMinusW = new DenseTensor1(model.D, 1)-hyperPlane
        val posScaled = posGrad
        posScaled *= oneMinusW
        val negScaled = negGrad
        negScaled *= oneMinusW

        gradient.accumulate(model.weights(e1PosDex), posScaled, factor)
        gradient.accumulate(model.weights(e2PosDex), posScaled, -factor)
        // dont scale
        gradient.accumulate(model.weights(relDex), posGrad - negGrad, factor)
        // scale by (1-w)
        gradient.accumulate(model.weights(e1NegDex), negScaled, -factor)
        gradient.accumulate(model.weights(e2NegDex), negScaled, factor)

        posGrad *= (e2PosEmb-e1PosEmb)
        negGrad *= (e2NegEmb-e1NegEmb)
        val hGradScaled = posGrad - negGrad
        // scale by (t-h) and (t'-h')
        gradient.accumulate(model.hyperPlanes(hPlaneDex), hGradScaled, factor)
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
  println(transH.avgRankHitsAt10(test.map(x => (x._2, x._4, x._3))))
  println(Evaluator.avgRankHitsAt10(transH, test))

}
