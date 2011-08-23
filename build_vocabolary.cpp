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
	string dir = "foodcamimages/TRAIN", filepath;
	DIR *dp;
	struct dirent *dirp;
	struct stat filestat;
	
	dp = opendir( dir.c_str() );
	
	// detecting keypoints
	SurfFeatureDetector detector(400);
	//FastFeatureDetector detector(1,true);
	vector<KeyPoint> keypoints;	
	
	// computing descriptors
	//Ptr<DescriptorExtractor > extractor(new SurfDescriptorExtractor());//  extractor;
	Ptr<DescriptorExtractor > extractor(
		new OpponentColorDescriptorExtractor(
			Ptr<DescriptorExtractor>(new SurfDescriptorExtractor())
			)
		);
	Mat descriptors;
	Mat training_descriptors(1,extractor->descriptorSize(),extractor->descriptorType());
	Mat img;
	
	cout << "------- build vocabulary ---------\n";
	
	cout << "extract descriptors.."<<endl;
	//int count = 0;
	Rect clipping_rect = Rect(0,120,640,480-120);
	Mat bg_ = imread("background.png")(clipping_rect), img_fg;
	while (dirp = readdir( dp ))
    {
	//	count++;
		filepath = dir + "/" + dirp->d_name;
		
		// If the file is a directory (or is in some way invalid) we'll skip it 
		if (stat( filepath.c_str(), &filestat )) continue;
		if (S_ISDIR( filestat.st_mode ))         continue;
		
		img = imread(filepath);
		if (!img.data) {
			continue;
		}
		img = img(clipping_rect);
		img_fg = img - bg_;
		detector.detect(img_fg, keypoints);
//		{
//			Mat out; //img_fg.copyTo(out);
//			drawKeypoints(img, keypoints, out, Scalar(255));
//			imshow("fg",img_fg);
//			imshow("keypoints", out);
//			waitKey(0);
//		}
		extractor->compute(img, keypoints, descriptors);
		
		training_descriptors.push_back(descriptors);
		cout << ".";
    }
	cout << endl;
	closedir( dp );
	
	cout << "Total descriptors: " << training_descriptors.rows << endl;
	
	FileStorage fs("training_descriptors.yml", FileStorage::WRITE);
	fs << "training_descriptors" << training_descriptors;
	fs.release();
    
	BOWKMeansTrainer bowtrainer(1000); //num clusters
	bowtrainer.add(training_descriptors);
	cout << "cluster BOW features" << endl;
	Mat vocabulary = bowtrainer.cluster();
	
	FileStorage fs1("vocabulary_color_1000.yml", FileStorage::WRITE);
	fs1 << "vocabulary" << vocabulary;
	fs1.release();
}
