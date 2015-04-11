package cc.factorie.app.nlp.embeddings

import cc.factorie.la.DenseTensor1
import scala.io.Source
import scala.collection.mutable.PriorityQueue


object DocDocDistance {

  var threshold = 0
  var docs = Array[String]()
  var weights = Array[DenseTensor1]()
  var D = 200

  var top = 30
  var docNum=100000
  def main(args: Array[String]) {

    if (args.size != 1) {
      println("Input vectors file missing. USAGE : distance vectors.txt")
      return
    }
    val inputFile = args(0)
    load(inputFile)
    play()
  }

  def nearestNeighbours(inputFile: String, numNeighbours: Int = 30): Unit = {
    top = numNeighbours
    load(inputFile)
    play()
  }
  def load(embeddingsFile: String): Unit = {
    var lineItr = Source.fromFile(embeddingsFile).getLines
    // first line is (# words, dimension)
    //val details = lineItr.next.stripLineEnd.split(' ').map(_.toInt)
    //V = if (threshold > 0 && details(0) > threshold) threshold else details(0)
    //D = details(1)
    //docNum=  lineItr.next.stripLineEnd.toInt
    println("docs "+docNum)
    lineItr.next
    docs = new Array[String](docNum)
    weights = new Array[DenseTensor1](docNum)
    for (v <- 0 until docNum) {
      val line = lineItr.next.stripLineEnd.split(' ')

      docs(v) = line(0)
      weights(v) = new DenseTensor1(D, 0) // allocate the memory
      for (d <- 0 until D) weights(v)(d) = line(d + 1).toDouble
      weights(v) /= weights(v).twoNorm
    }
    println("loaded document vocab and their embeddings")
  }
  def play(): Unit = {

    while (true) {
      print("Enter docId (EXIT to break) : ")
      var words = readLine.stripLineEnd.toLowerCase.split(' ').map(word => getdocID(word)).filter(id => id != -1)
      if (words.size == 0) {
        println("words not in vocab")
      } else {
        val embedding_in = new DenseTensor1(D, 0)
        words.foreach(word => embedding_in.+=(weights(word)))

        embedding_in./=(words.size)
        var pq = new PriorityQueue[(String, Double)]()(dis)
        //println(docs.size)
        for (i <- 0 until docs.size) {
          val embedding_out = weights(i)
          val score = TensorUtils.cosineDistance(embedding_in, embedding_out)
          //println(score)
          //println(score)
          if (i < top) pq.enqueue(docs(i) -> score)
          else if (score > pq.head._2) { // if the score is greater the min, then add to the heap
            pq.dequeue
            pq.enqueue(docs(i) -> score)
          }
        }
        var arr = new Array[(String, Double)](pq.size)
        var i = 0
        while (!pq.isEmpty) { // min heap
          arr(i) = (pq.head._1, pq.head._2)
          i += 1
          pq.dequeue
        }
        println("\t\t\t\t\t\tWord\t\tCosine Distance")
        arr.reverse.foreach(x => println("%50s\t\t%f".format(x._1, x._2)))

      }
    }
  }
  // private helper functions
  private def dis() = new Ordering[(String, Double)] {
    def compare(a: (String, Double), b: (String, Double)) = -a._2.compare(b._2)
  }
  /*private def getID(word: String): Int = {
    for (i <- 0 until vocab.length) if (vocab(i).equalsIgnoreCase(word))
      return i
    return -1

  } */
  private def getdocID(id: String): Int = {
    for (i <- 0 until docs.length) if (docs(i).equals(id))
      return i
    return -1

  }

}

