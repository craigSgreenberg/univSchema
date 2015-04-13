#!/bin/bash
# run_class.sh [<jvm options>] <scala object with main> [<arguments>]

args="$@"

memory=120g

# If the root directory is not passed as an environment variable, set it to 
# the same dir as this script.
if [ -z "$ROOT_DIR" ]
then
  ROOT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
fi

# Get classpath: all jar files from maven -- either from a file CP.hack, or 
# if this file does not exist yet, from the output of a maven compile.
# TODO: there should be a nice and automated way for starting after building -
# for now, this hack attempts that, somehow.
if [ ! -f "$ROOT_DIR/CP.hack" ]
then
 echo 'CP.hack does not exist, get class paths ...'
 cd $ROOT_DIR
 mvn compile -X \
 | grep 'classpathElements = ' \
 | sed 's#^.* classpathElements = \[\(.*\)\]$#\1#g' \
 | sed 's#, #:#g' \
 | head -1 \
 > $ROOT_DIR/CP.hack
 cd -
 echo '... done'
fi

CP=`cat $ROOT_DIR/CP.hack`

#jacksonjars=${HOME}/.m2/repository/com/fasterxml/jackson/core/jackson-core/2.2.2/jackson-core-2.2.2.jar:${HOME}/.m2/repository/com/fasterxml/jackson/core/jackson-databind/2.2.2/jackson-databind-2.2.2.jar:${HOME}/.m2/repository/com/fasterxml/jackson/core/jackson-annotations/2.2.2/jackson-annotations-2.2.2.jar:${HOME}/.m2/repository/com/fasterxml/jackson/module/jackson-module-scala_2.10/2.2.2/jackson-module-scala_2.10-2.2.2.jar
#factoriejar=${HOME}/.m2/repository/cc/factorie/factorie/1.1-TAC-SNAPSHOT/factorie-1.1-TAC-SNAPSHOT.jar
#modelsjar=${HOME}/.m2/repository/cc/factorie/app/nlp/all-models/1.0-SNAPSHOT/all-models-1.0-SNAPSHOT.jar
#scalajar=${HOME}/.m2/repository/org/scala-lang/scala-library/2.10.2/scala-library-2.10.2.jar
#scalareflect=${HOME}/.m2/repository/org/scala-lang/scala-reflect/2.10.2/scala-reflect-2.10.2.jar
#CP=$jacksonjars:$factoriejar:$modelsjar:$scalajar:$scalareflect:${ROOT_DIR}/target/tackbp2014-0.1-SNAPSHOT.jar

#echo "using classpath: $CP"

java -Dfile.encoding=UTF8 -cp $CP -Xmx$memory $args

