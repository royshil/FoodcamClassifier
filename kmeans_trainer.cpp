/*
 *  kmeans_trainer.cpp
 *  FoodcamClassifier
 *
 *  Created by Roy Shilkrot on 8/20/11.
 *  Copyright 2011 MIT. All rights reserved.
 *
 */

#include "kmeans_trainer.h"

int main(int argc, char** argv) {

	FileStorage fs("training _descriptors.yml", FileStorage::READ);
	Mat training_descriptors;
	fs["training_descriptors"] >> training_descriptors;
	fs.release();
	
	BOWKMeansTrainer bowtrainer(1000); //num clusters
	bowtrainer.add(training_descriptors);
	cout << "cluster BOW features" << endl;
	Mat vocabulary = bowtrainer.cluster();

	FileStorage fs1("vocabulary_1000.yml", FileStorage::WRITE);
	fs1 << "vocabulary" << vocabulary;
	fs1.release();

}