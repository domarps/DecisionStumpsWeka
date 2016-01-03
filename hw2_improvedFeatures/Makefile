JFLAGS = -cp lib/weka.jar:bin -d bin -sourcepath src
JAVAC = javac

.PHONY: directories
.SUFFIXES: .java .class

.java.class:
	$(JAVAC) $(JFLAGS) $*.java

CLASSES =  src/cs446/weka/classifiers/trees/FeatureGenerator.java  src/cs446/weka/classifiers/trees/Id3.java  src/cs446/weka/classifiers/trees/WekaTester.java  src/cs446/weka/classifiers/trees/SGDTester.java src/cs446/weka/classifiers/trees/SGD.java src/cs446/weka/classifiers/trees/Id3Tester.java src/cs446/weka/classifiers/trees/Id3DecStump4Tester.java src/cs446/weka/classifiers/trees/Id3DecStump8Tester.java src/cs446/weka/classifiers/trees/Id3StumpSGDTester.java

all: directories classes

directories: 
	mkdir -p bin

classes: $(CLASSES:.java=.class)

clean:
	$(RM) -r bin

