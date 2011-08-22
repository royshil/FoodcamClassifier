/*
 *  training_common.cpp
 *  FoodcamClassifier
 *
 *  Created by Roy Shilkrot on 8/22/11.
 *  Copyright 2011 MIT. All rights reserved.
 *
 */

#include "train_bovw.h"

void trainSVM(map<string,Mat>& classes_training_data, string& file_postfix, int response_cols, int response_type) {

	//train 1-vs-all SVMs
	map<string,CvSVM> classes_classifiers;
	for (map<string,Mat>::iterator it = classes_training_data.begin(); it != classes_training_data.end(); ++it) {
		string class_ = (*it).first;
		cout << "training class: " << class_ << ".." << endl;
		
		Mat samples(0,response_cols,response_type);
		Mat labels(0,1,CV_32FC1);
		
		//copy class samples and label
		cout << "adding " << classes_training_data[class_].rows << " positive"<<endl;
		samples.push_back(classes_training_data[class_]);
		Mat class_label = Mat::ones(classes_training_data[class_].rows, 1, CV_32FC1);
		labels.push_back(class_label);
		
		//copy rest samples and label
		for (map<string,Mat>::iterator it1 = classes_training_data.begin(); it1 != classes_training_data.end(); ++it1) {
			string not_class_ = (*it1).first;
			if(not_class_[0] == class_[0]) continue;
			samples.push_back(classes_training_data[not_class_]);
			class_label = Mat::zeros(classes_training_data[not_class_].rows, 1, CV_32FC1);
			labels.push_back(class_label);
		}
		
		Mat samples_32f; samples.convertTo(samples_32f, CV_32F);
		if(samples.rows == 0) continue; //phantom class?!
		classes_classifiers[class_].train(samples_32f,labels);
		
		stringstream ss; 
		ss << "SVM_classifier_"; 
		if(file_postfix.size() > 0) ss << file_postfix << "_";
		ss << class_ << ".yml";
		classes_classifiers[class_].save(ss.str().c_str());
	}
}

void extract_training_samples(Ptr<FeatureDetector>& detector, BOWImgDescriptorExtractor& bowide, map<string,Mat>& classes_training_data) {
	Mat response_hist;
	cout << "look in train data"<<endl;
	char buf[255];
	ifstream ifs("training.txt");
	int total_samples = 0;
	string filepath;
	Mat img;
	vector<KeyPoint> keypoints;
	vector<string> classes_names;
	do
	{
		ifs.getline(buf, 255);
		string line(buf);
		istringstream iss(line);
		//		cout << line << endl;
		iss >> filepath;
		Rect r; char delim; iss >> r.x >> delim >> r.y >> delim >> r.width >> delim >> r.height;
		string class_; iss >> class_;
		
		img = imread(filepath);
		r &= Rect(0,0,img.cols,img.rows);
		if(r.width != 0) {
			img = img(r); //crop to interesting region
		}
		cout << "."; cout.flush();
		//		char c__[] = {(char)atoi(class_.c_str()),'\0'};
		//		string c_(c__);
		//		cout << c_; cout.flush();
		//		putText(img, c_, Point(20,20), CV_FONT_HERSHEY_PLAIN, 2.0, Scalar(255), 2);
		//		imshow("pic",img);
		
		detector->detect(img,keypoints);
		bowide.compute(img, keypoints, response_hist);
		
		if(classes_training_data.count(class_) == 0) { //not yet created...
			classes_training_data[class_].create(0,response_hist.cols,response_hist.type());
			classes_names.push_back(class_);
		}
		classes_training_data[class_].push_back(response_hist);
		total_samples++;
		
		//		waitKey(0);
	} while (!ifs.eof());
	cout << endl;

	cout << "save to file.."<<endl;
	{
		FileStorage fs("training_samples.yml",FileStorage::WRITE);
		fs << "classes" << classes_names;
		for (map<string,Mat>::iterator it = classes_training_data.begin(); it != classes_training_data.end(); ++it) {
			fs << (*it).first << (*it).second;
		}
	}
}