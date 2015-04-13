#!/bin/bash
#
# Build event tagging stuff
#

git pull
mvn -Dmaven.test.skip=true package
