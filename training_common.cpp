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
	cout << "look in train data"<<endl;
	char buf[255];
	Ptr<ifstream> ifs(new ifstream("training.txt"));
	int total_samples = 0;
	string filepath;
	vector<string> classes_names;
	
	vector<string> lines;
	while(!ifs->eof()) {
		ifs->getline(buf, 255);
		lines.push_back(buf);
	}
	
	//try some multithreading
	#pragma omp parallel for
	for(int i=0;i<lines.size();i++) {
//		printf("Hello from thread %d, nthreads %d\n", omp_get_thread_num(), omp_get_num_threads());
//		if(ifs->eof()) break;

		vector<KeyPoint> keypoints;
		Mat response_hist;
		Mat img;

		string line(lines[i]);
		istringstream iss(line);

		iss >> filepath;
		Rect r; char delim; iss >> r.x >> delim >> r.y >> delim >> r.width >> delim >> r.height;
		string class_; iss >> class_;
		
		img = imread(filepath);
		r &= Rect(0,0,img.cols,img.rows);
		if(r.width != 0) {
			img = img(r); //crop to interesting region
		}
		//		char c__[] = {(char)atoi(class_.c_str()),'\0'};
		//		string c_(c__);
		//		cout << c_; cout.flush();
		//		putText(img, c_, Point(20,20), CV_FONT_HERSHEY_PLAIN, 2.0, Scalar(255), 2);
		//		imshow("pic",img);
		
		detector->detect(img,keypoints);
		bowide.compute(img, keypoints, response_hist);

		cout << "."; cout.flush();

		#pragma omp critical
		{
			if(classes_training_data.count(class_) == 0) { //not yet created...
				classes_training_data[class_].create(0,response_hist.cols,response_hist.type());
				classes_names.push_back(class_);
			}
			classes_training_data[class_].push_back(response_hist);
		}
		total_samples++;

		//		waitKey(0);
	}
	cout << endl;

	cout << "save to file.."<<endl;
	{
		FileStorage fs("training_samples.yml",FileStorage::WRITE);
		for (map<string,Mat>::iterator it = classes_training_data.begin(); it != classes_training_data.end(); ++it) {
			fs << (*it).first << (*it).second;
		}
	}
}