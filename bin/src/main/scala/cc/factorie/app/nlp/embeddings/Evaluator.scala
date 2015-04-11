/* Copyright (C) 2008-2014 University of Massachusetts Amherst.
   This file is part of "FACTORIE" (Factor graphs, Imperative, Extensible)
   http://factorie.cs.umass.edu, http://github.com/factorie
   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0
   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License. */
package cc.factorie.app.nlp.embeddings
import java.nio.charset.Charset
import scala.collection.mutable.ArrayBuffer


object Evaluator {
  def meanAveragePrecision(classToPredictionAndLabel: scala.collection.mutable.Map[Int, ArrayBuffer[(Double, Boolean)]]): Double = {
    classToPredictionAndLabel.values.map(averagePrecision(_)).sum / classToPredictionAndLabel.size.toDouble
  }

  def averagePrecision(predictionAndLabel: ArrayBuffer[(Double, Boolean)]): Double = {
    val judgements = predictionAndLabel.sortBy(-_._1).map(_._2)
    // This foldleft aggregates (#true, #false, averagePrecision).
    // #true and #false are simply counted up.
    // averagePrecision (ap) is updated for true items:
    // ap_new = (#true_new - 1) * ap_old / #true_new + 1 / (#true_new + #false_new)
    //        = (#true_old) * ap_old / (#true_old + 1) + 1 / (#true_old + #false_old + 1)
    val mapStats = judgements.foldLeft((0,0,0.0))((stat,j) => {if (j==true) {
      (stat._1 + 1, stat._2, stat._1 * stat._3 / (stat._1 + 1) + 1.0/(stat._1 + stat._2 + 1))
    } else {
      (stat._1, stat._2 + 1, stat._3)
    }})
    mapStats._3
  }

  // convenience method
  def averagePrecision(classToPredictionAndLabel: scala.collection.mutable.Map[Int, Seq[(Double, Boolean)]]): Double = {
    throw new UnsupportedOperationException
  }
}
