/*
 *  predict_common.h
 *  FoodcamClassifier
 *
 *  Created by Roy Shilkrot on 8/23/11.
 *  Copyright 2011 MIT. All rights reserved.
 *
 */


#include <stdio.h>
#include <stdlib.h>

#include <opencv2/opencv.hpp>
#include <fstream>
#include <iostream>
#include <string>
#include <set>

#include <dirent.h>
#include <unistd.h>
#include <sys/stat.h>
#include <sys/types.h>

#include <omp.h>

using namespace cv;
using namespace std;

class FoodcamPredictor {
public:	
	FoodcamPredictor();
	void evaluateOneImage(Mat& __img, vector<string>& out_classes);
	map<string,CvSVM>& getClassesClassifiers() { return classes_classifiers; }
	void normalizeClassname(string& max_class) { 
		if(max_class.compare("cake")==0) max_class = "cookies";
		if(max_class.compare("fruit")==0) max_class = "fruit_veggie";
	}
	void setDebug(bool _b) { debug = _b;}
	
private:
	void initColors();
	void initSVMs();
	void initVocabulary();
	
	bool debug;
	
	Ptr<FeatureDetector > detector;
	Ptr<BOWImgDescriptorExtractor > bowide;
	Ptr<DescriptorMatcher > matcher;
	Ptr<DescriptorExtractor > extractor;
	map<string,CvSVM> classes_classifiers;
	map<string,Scalar> classes_colors;
	Mat background;
	Mat vocabulary;
};