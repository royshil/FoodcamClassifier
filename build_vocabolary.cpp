/*
 *  build_vocabolary.cpp
 *  FoodcamClassifier
 *
 *  Created by Roy Shilkrot on 8/19/11.
 *  Copyright 2011 MIT. All rights reserved.
 *
 */

#include "build_vocabolary.h"



int main(int argc, char** argv) {
	string dir = "/Users/royshilkrot/Downloads/foodcamimages/TRAIN", filepath;
	DIR *dp;
	struct dirent *dirp;
	struct stat filestat;
	
	dp = opendir( dir.c_str() );
	
	// detecting keypoints
	SurfFeatureDetector detector(1000);
	vector<KeyPoint> keypoints;	
	
	// computing descriptors
	Ptr<DescriptorExtractor > extractor(new SurfDescriptorExtractor());//  extractor;
	Mat descriptors;
	Mat training_descriptors(1,extractor->descriptorSize(),extractor->descriptorType());
	Mat img;
	
	cout << "------- build vocabulary ---------\n";
	
	cout << "extract descriptors.."<<endl;
	//int count = 0;
	while (dirp = readdir( dp ))
    {
	//	count++;
		filepath = dir + "/" + dirp->d_name;
		
		// If the file is a directory (or is in some way invalid) we'll skip it 
		if (stat( filepath.c_str(), &filestat )) continue;
		if (S_ISDIR( filestat.st_mode ))         continue;
		
		img = imread(filepath);
		detector.detect(img, keypoints);
		extractor->compute(img, keypoints, descriptors);
		
		training_descriptors.push_back(descriptors);
		cout << ".";
    }
	cout << endl;
	closedir( dp );
	
	cout << "Total descriptors: " << training_descriptors.rows << endl;
	
	FileStorage fs("training _descriptors.yml", FileStorage::WRITE);
	fs << "training_descriptors" << training_descriptors;
	fs.release();
    
	BOWKMeansTrainer bowtrainer(1000); //num clusters
	bowtrainer.add(training_descriptors);
	cout << "cluster BOW features" << endl;
	Mat vocabulary = bowtrainer.cluster();
	
	FileStorage fs1("vocabulary_1000.yml", FileStorage::WRITE);
	fs1 << "vocabulary" << vocabulary;
	fs1.release();
}