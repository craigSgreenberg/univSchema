package cc.factorie.app.nlp.embeddings

import cc.factorie.optimize.{AdaGradRDA, Example}
import cc.factorie.util.{Threading, DoubleAccumulator}
import cc.factorie.la._
import java.util.zip.GZIPInputStream
import java.io.{File, PrintWriter, FileInputStream}
import scala.collection.mutable.ArrayBuffer
import cc.factorie.model.Weights
import java.util

class NeighborhoodClassifier (override val opts: EmbeddingOpts) extends UniversalSchemaModel(opts) {


  override def buildVocab(): Unit ={

    val  testcorpusLineItr = io.Source.fromInputStream(new FileInputStream(opts.testRelationsFile.value), encoding).getLines
    while (testcorpusLineItr.hasNext) {
      val rel = testcorpusLineItr.next.stripLineEnd
      //val Array(ep, rel, label) = line.stripLineEnd.split('\t')
      //val relKey = relationKey.getOrElseUpdate(rel, relationKey.size)
      if(!(relationKey.containsKey(rel))) relationKey.put(rel, relationKey.size())
      testRels.add(rel)
    }

    println("Number of test relations ", testRels.size)
    val examples = new ArrayBuffer[(Int,Int, Int, Int)]()

    def ingestCorpus(thisCorpus:String, relMap:util.HashMap[String, Int], isLabelSpace:Boolean, startIndex:Int):Int = {
      println(thisCorpus)
      println(encoding)
      val corpusLineItr = thisCorpus.endsWith(".gz") match {
        //case true => io.Source.fromInputStream(new GZIPInputStream(new FileInputStream(thisCorpus)), encoding).getLines
        //case false => io.Source.fromInputStream(new FileInputStream(thisCorpus), encoding).getLines
        case true => io.Source.fromInputStream(new GZIPInputStream(new FileInputStream(thisCorpus)), "UTF-16").getLines
        case false => io.Source.fromInputStream(new FileInputStream(thisCorpus), "UTF-16").getLines
      }
      while (corpusLineItr.hasNext) {
        val line = corpusLineItr.next
        val Array(ep, rel, label) = line.stripLineEnd.split('\t')
        if(!(entPairKey.containsKey(ep)))  entPairKey.put(ep, entPairKey.size())
        //if(!(relationKey.containsKey(rel))) relationKey.put(rel, relationKey.size())
        if(!(relMap.containsKey(rel))) relMap.put(rel, relMap.size())
        val epKey:Int = entPairKey.get(ep)
        //val Array(e1, e2) = ep.split(",")
        //val e1Key = entityVocab.get(e1)
        //val e2Key = entityVocab.get(e2)
        val relKey:Int = startIndex + relMap.get(rel)
        if(isLabelSpace && testRels.contains(rel)) examples += ((epKey, 0, 0, relKey))
        entityPairFeatures(epKey) = entityPairFeatures.getOrElseUpdate(epKey, new SparseTensor1(200000))
        entityPairFeatures(epKey).update(relKey, label.toFloat)
      }
      startIndex + relMap.size()
    }

    var numDim = 0
    if (!opts.corpus.value.isEmpty) numDim += ingestCorpus(corpus, relationKey, isLabelSpace = true, numDim)
    if (!opts.freebaseWordFeatures.value.isEmpty) numDim += ingestCorpus(opts.freebaseWordFeatures.value, new util.HashMap[String, Int](), isLabelSpace = false, numDim)
    if (!opts.wikiWordFeatures.value.isEmpty) numDim += ingestCorpus(opts.wikiWordFeatures.value, new util.HashMap[String, Int](), isLabelSpace = false, numDim)
    trainingExamples = examples.toSeq
    entPairSize = entPairKey.size
    relationSize = relationKey.size
    trainingExamplesSize = trainingExamples.size
    println("Number of entity pairs: ", entPairSize)
    println("Number of relations: ", relationSize)
    println("Number of training examples: ", trainingExamplesSize)
  }

  override def learnEmbeddings(): Unit = {

    println("Learning Classifier")
    optimizer = new AdaGradRDA(delta = adaGradDelta, rate = adaGradRate, l2 = opts.regularizer.value)
    //weights = (0 until entPairSize).map(i => Weights(TensorUtils.setToRandom1(new DenseTensor1(D, 0), rand))) // initialized using wordvec random
    // for hierarchical softmax
    nodeWeights =  (0 until testRels.size).map(i => Weights(TensorUtils.setToRandom1(new DenseTensor1(relationSize, 0), rand)))
    // set for paragraph vector
    //parWeights =   (0 until docNum).map(i => Weights(TensorUtils.setToRandom1(new DenseTensor1(D, 0))))
    optimizer.initializeWeights(this.parameters)
    trainer = new LiteHogwildTrainer(weightsSet = this.parameters, optimizer = optimizer, nThreads = threads, maxIterations = Int.MaxValue)
    println("number of threads: ", threads)
    val threadIds = (0 until threads).map(i => i)
    //var word_count: Long = 0
    val groupSize = trainingExamplesSize % threads == 0 match {
      case true => trainingExamplesSize/threads
      case false =>  trainingExamplesSize/threads + 1
    }
    for(i <- 1 until (opts.epochs.value + 1)){ // number of iterations
      println("Training Iteration " , i , processed)
      processed = 0
      val it = rand.shuffle(trainingExamples).grouped(groupSize);
      //println("match ", threads, groupSize)
      var threadExamples = new ArrayBuffer[Seq[(Int, Int, Int, Int)]]()
      for(n <- 0 until threads)  threadExamples = threadExamples += it.next()
      Threading.parForeach(threadIds, threads)(threadId => workerThread(threadExamples(threadId)))
      if(i % opts.evalautionFrequency.value == 0) {
        println("Dev MAP after " + i + " iterations: " + evaluate(opts.devFile.value, i))
        println("Test MAP after " + i + " iterations: " + evaluate(opts.testFile.value, i))
      }
    }
    println("Done training classifiers. ")
  }


  override def process(ep: Int, rel: Int): Unit = {
        trainer.processExample(new NeighborhoodClassifierExample(this, ep, rel))
  }

  override def getScore(ep: Int, rel: Int): Double = {
    //nodeWeights(rel).value.dot(entityPairFeatures(ep))
    var ans = 0.0

    //val nodeWeight = nodeWeights(rel).value.toArray
    val nodeWeight = nodeWeights(rel).value
    val features = entityPairFeatures(ep)
    /*
    for(features.foreachActiveElement()){
      if(f != rel)  ans += nodeWeight(f)
    }
    */
    //features.foreachActiveElement((t : (Int, Double)) => ans += nodeWeight(t._1))
    //println("example ", ep, rel)
    features.foreachActiveElement({case(index,value) => {if(index != rel) { ans +=nodeWeight(index)}}})
    ans
  }
}

class NeighborhoodClassifierExample(model: UniversalSchemaModel, ep: Int, rel:Int) extends Example {

  def getNegEp(): Int = {
    model.rand.nextInt(model.entPairSize)

  }

  def accumulateValueAndGradient(value: DoubleAccumulator, gradient: WeightsMapAccumulator): Unit = {

    for(j <- 0 until model.opts.negative.value) {
      val negEp: Int = getNegEp()
      if(negEp == -1) return
      model.processed += 1
      val epFeatures =   model.entityPairFeatures(ep)
      val negEpFeatures: SparseTensor1 =   model.entityPairFeatures(negEp)

      {
        //+ve example
        val score: Double = model.getScore(ep, rel)
        val exp: Double = math.exp(-score)
        var objective: Double = 0.0
        var factor: Double = 0.0
        objective = -math.log1p(exp)
        factor = exp / (1 + exp)
        if (value ne null) value.accumulate(objective)
        if (gradient ne null) {
          gradient.accumulate(model.nodeWeights(rel), epFeatures, factor)
        }
        0
      }
      {
        //-ve example
        val score: Double = model.getScore(negEp, rel)
        val exp: Double = math.exp(-score)
        var objective: Double = 0.0
        var factor: Double = 0.0
        objective = -score - math.log1p(exp)
        factor = -1 / (1 + exp)
        if (value ne null) value.accumulate(objective)
        if (gradient ne null) {
          gradient.accumulate(model.nodeWeights(rel), negEpFeatures, factor)
        }
        0
      }
    }
  }
}