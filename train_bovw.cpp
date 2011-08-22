/*
 *  train_bovw.cpp
 *  FoodcamClassifier
 *
 *  Created by Roy Shilkrot on 8/19/11.
 *  Copyright 2011 MIT. All rights reserved.
 *
 */

#include "train_bovw.h"

int main(int argc, char** argv_) {
	vector<string> argv;for (int i = 0; i < argc; i++) {argv.push_back(argv_[i]);}
	
	if (argv.size() == 1) {
		cout << "USAGE: train_bovw <vocabulary_file.yml> [prefix_for_output]"<<endl;
		return 1;
	}
	
	cout << "-------- train BOVW SVMs -----------" << endl;
	cout << "read vocabulary form file"<<endl;
	Mat vocabulary;
	FileStorage fs(argv[1], FileStorage::READ);
	fs["vocabulary"] >> vocabulary;
	fs.release();	
	
	Ptr<SurfFeatureDetector > detector(new SurfFeatureDetector()); //detector
	//Ptr<DescriptorExtractor > extractor(new SurfDescriptorExtractor());//  extractor;
	Ptr<DescriptorExtractor > extractor(
		new OpponentColorDescriptorExtractor(
			 Ptr<DescriptorExtractor>(new SurfDescriptorExtractor())
			 )
		);
	Ptr<DescriptorMatcher > matcher(new BruteForceMatcher<L2<float> >());
	BOWImgDescriptorExtractor bowide(extractor,matcher);
	bowide.setVocabulary(vocabulary);
	
	//setup training data for classifiers
	map<string,Mat> classes_training_data; classes_training_data.clear();
	
	cout << "train SVMs\n";
	
	Mat response_hist;
	cout << "look in train data"<<endl;
//	int count = 0;
	char buf[255];
	ifstream ifs("training.txt");
	int total_samples = 0;
	string filepath;
	Mat img;
	vector<KeyPoint> keypoints;
	do
    {
		ifs.getline(buf, 255);
		string line(buf);
		istringstream iss(line);
//		cout << line << endl;
		iss >> filepath;
		Rect r; char delim;
		iss >> r.x >> delim;
		iss >> r.y >> delim;
		iss >> r.width >> delim;
		iss >> r.height;
		//		cout << r.x << "," << r.y << endl;
		string class_;
		iss >> class_;
		
		img = imread(filepath);
		r &= Rect(0,0,img.cols,img.rows);
		if(r.width != 0) {
			img = img(r); //crop to interesting region
		}
		char c__[] = {(char)atoi(class_.c_str()),'\0'};
		string c_(c__);
		cout << c_;
//		putText(img, c_, Point(20,20), CV_FONT_HERSHEY_PLAIN, 2.0, Scalar(255), 2);
//		imshow("pic",img);
		detector->detect(img,keypoints);
		bowide.compute(img, keypoints, response_hist);
		
		if(classes_training_data.count(c_) == 0) { //not yet created...
			classes_training_data[c_].create(0,response_hist.cols,response_hist.type());
		}
		classes_training_data[c_].push_back(response_hist);
		total_samples++;
//		waitKey(0);
	} while (!ifs.eof());
	cout << endl;
	
	cout << "got " << classes_training_data.size() << " classes.\n total of " << total_samples << " samples." <<endl;
	for (map<string,Mat>::iterator it = classes_training_data.begin(); it != classes_training_data.end(); ++it) {
		cout << " class " << (*it).first << " has " << (*it).second.rows << " samples"<<endl;
	}
	
	//train 1-vs-all SVMs
	map<string,CvSVM> classes_classifiers;
	for (map<string,Mat>::iterator it = classes_training_data.begin(); it != classes_training_data.end(); ++it) {
		string class_ = (*it).first;
		cout << "training class: " << class_ << ".." << endl;
		
		Mat samples(0,response_hist.cols,response_hist.type());
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
		if(argv.size() > 2) ss << argv[3] << "_";
		ss << class_ << ".yml";
		classes_classifiers[class_].save(ss.str().c_str());
	}
	
	return 0;
}
