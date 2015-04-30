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
import cc.factorie.util.CmdOptions

class EmbeddingOpts extends CmdOptions {

  // Algorithm related
  val epochs = new CmdOption("epochs", 200, "INT", "training epochs")
  val evalautionFrequency = new CmdOption("frequency", 10, "INT", "evaluation frequency")
  val dimension = new CmdOption("size", 200, "INT", "use <int> size of word vectors")
  val window = new CmdOption("window", null, "INT", "use <int> skip length between words")
  val threads = new CmdOption("threads", 20, "INT", "use <int> threads")
  val negative = new CmdOption("negative", 1, "INT", "use <int> number of negative examples")
  val minCount = new CmdOption("min-count", null, "INT", "This will discard words that appear less than <int> times; default is 5")
  val ignoreStopWords = new CmdOption("ignore-stopwords", false, "BOOLEAN", "use <bool> to include or discard stopwords. Use 1 for discarding stopwords")
  val cbow = new CmdOption("cbow", false, "BOOLEAN", "user cbow=true for cbow and cbow=false for skip-gram") // 1 would be SkipGram // default method is skipgram
  val writeOutput = new CmdOption("writeOutput", false, "BOOLEAN", "writing test and dev file predictions")
  val hinge = new CmdOption("hinge", false, "BOOLEAN", "will use hinge instead of sigmoid")
  val margin = new CmdOption("margin", 0.0, "DOUBLE", "margin for hinge")
  val wsabie = new CmdOption("wsabie", false, "BOOLEAN", "sample till you get a -ve > +ve")
  val sample = new CmdOption("sample", null, "DOUBLE", "use <double> subsampling")
  val hierSoftMax = new CmdOption("hier-soft-max", true, "BOOLEAN", "true if hierarchical softmax is used for training, else use negative sampling")
  val options = new CmdOption("options", 2, "INT", "1-HierarchicalSoftmax, 2-NegativeSampling, 3-NeighborhoodClassifier")
  val writeVecs = new CmdOption("writeVecs", false, "BOOLEAN", "writing relations vecs")
  // Optimization related (Don't change if you do not understand how vectors are initialized)
  val rate = new CmdOption("rate", 0.1, "DOUBLE", "learning rate for adaGrad")
  val regularizer = new CmdOption("regularizer", 0.01, "DOUBLE", "learning rate for adaGrad")
  val delta = new CmdOption("delta", 0.01, "DOUBLE", "delta for adaGrad")

  // IO Related (MUST GIVE Options)
  val encoding = new CmdOption("encoding", "UTF8", "STRING", "use <string> for encoding option. ISO-8859-15 is default")
  val saveVocabFile = new CmdOption("save-vocab", null, "STRING", "save vocab file")
  val loadVocabFile = new CmdOption("load-vocab", null, "STRING", "load the vocab file") // atleast one of them  should be given. save-vocab or load-vocab
  val corpus = new CmdOption("train", "", "STRING", "train file")
  val freebaseWordFeatures = new CmdOption("free-word", "", "STRING", "freebase word features file")
  val wikiWordFeatures = new CmdOption("wiki-word", "", "STRING", "wikipedia word features file")
  val devFile = new CmdOption("dev", "", "STRING", "dev file")
  val testFile = new CmdOption("test", "", "STRING", "test file")
  val testRelationsFile = new CmdOption("testRelationsFile", "", "STRING", "test relations file")
  val treeFile = new CmdOption("tree", "", "STRING", "tree file")
  val output = new CmdOption("output", "", "STRING", "Use <file> to save the resulting word vectors")
  val binary = new CmdOption("binary", false, "BOOLEAN", "use true for storing .gz format and false for plain txt format. Both stores in ISO-8859-15 Encoding")
  val outputParagraph = new CmdOption("output-para", null, "STRING", "Use <file> to save the resulting paragraph vectors")

  // Vocabulary related
  // Maximum 14.3M * 0.7 = 10M words in the vocabulary (Don;t change if you understand how vocabBuilder works)
  val vocabSize = new CmdOption("max-vocab-size", null, "INT", "Max Vocabulary Size. Default Value is 2M . Reduce to 200k or 500k is you learn embeddings on small-data-set")
  val vocabHashSize = new CmdOption("vocab-hash-size", null, "INT", "Vocabulary hash size")
  val batchSize = new CmdOption[Int]("batch-size", 1200, "INT", "Size of each mini batch")
  val bernoulliSample = new CmdOption[Boolean]("bernoulli", false, "BOOLEAN", "Use bernoulli negative sampling, uniform otherwise.")
  val samplingTableSize = new CmdOption("sampling-table-size", null, "INT", "Sampling Table size")
  val l1 = new CmdOption[Boolean]("l1", true, "BOOLEAN", "Use l1 distance, l2 otherwise")
  val parseTsv = new CmdOption[Boolean]("parseTsv", false, "BOOLEAN", "Tsv formated training files")
  val featureSize = new CmdOption[Int]("feature-size", 10000000, "INT", "Max Feature Space Dimensionality")
}