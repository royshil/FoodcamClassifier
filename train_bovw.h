/*
 *  train_bovw.h
 *  FoodcamClassifier
 *
 *  Created by Roy Shilkrot on 8/19/11.
 *  Copyright 2011 MIT. All rights reserved.
 *
 */


#include <stdio.h>
#include <stdlib.h>

#include <opencv2/opencv.hpp>
#include <fstream>
#include <iostream>
#include <string>

#include <dirent.h>
#include <unistd.h>
#include <sys/stat.h>
#include <sys/types.h>

#include <omp.h>


using namespace cv;
using namespace std;

void trainSVM(map<string,Mat>& classes_training_data, string& file_postfix, int response_cols, int response_type);
void extract_training_samples(Ptr<FeatureDetector>& detector, BOWImgDescriptorExtractor& bowide, map<string,Mat>& classes_training_data);
