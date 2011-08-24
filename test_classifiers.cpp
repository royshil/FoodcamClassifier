/*
 *  test_classifiers.cpp
 *  FoodcamClassifier
 *
 *  Created by Roy Shilkrot on 8/20/11.
 *  Copyright 2011 MIT. All rights reserved.
 *
 */

#include "test_classifiers.h"

int main(int argc, char** argv) {
	string filepath;
	
	cout << "------- test ---------\n";
		
	ifstream ifs("test.txt",ifstream::in);
	char buf[255];
	vector<string> lines; 
	while(!ifs.eof()) {// && count++ < 30) {
		ifs.getline(buf, 255);
		lines.push_back(buf);
	}	
	ifs.close();
	cout << "total " << lines.size() << " samples to scan" <<endl;
	
	FoodcamPredictor predictor;
	predictor.setDebug(true);
	
	map<string,CvSVM>& classes_classifiers = predictor.getClassesClassifiers();
	map<string,map<string,int> > confusion_matrix;
	for (map<string,CvSVM>::iterator it = classes_classifiers.begin(); it != classes_classifiers.end(); ++it) {
		for (map<string,CvSVM>::iterator it1 = classes_classifiers.begin(); it1 != classes_classifiers.end(); ++it1) {
			string class1 = ((*it).first.compare("cake")==0) ? "cookies" : ((*it).first.compare("fruit")==0) ? "fruit_veggie" : (*it).first;
			string class2 = ((*it1).first.compare("cake")==0) ? "cookies" : ((*it1).first.compare("fruit")==0) ? "fruit_veggie" : (*it1).first;
			confusion_matrix[class1][class2] = 0;
		}
	}
	
	for (int i = 0; i < lines.size(); i++) {
		string line(lines[i]);
		cout << line << endl;
		istringstream iss(line);
		
		iss >> filepath;
//		Rect r; char delim; iss >> r.x >> delim >> r.y >> delim >> r.width >> delim >> r.height;
		vector<string> classes_; 
		while (!iss.eof()) {
			string class_; iss >> class_;
			classes_.push_back(class_);
		}
		
		if(classes_.size() == 0) continue;
		
		cout << "eval file " << filepath << " (" << i << "/" << lines.size() << ")" << endl;
		
		Mat __img = imread(filepath),_img;
		if(__img.size() != Size(640,480)) continue;
		vector<string> max_class;
		predictor.evaluateOneImage(__img,max_class);
		cout << "manual class: "; for(int j_=0;j_<classes_.size();j_++) cout << classes_[j_] << ",";
		cout << endl;
		
		int j_=0;
		for(;j_<classes_.size();j_++) {
			if(classes_[j_].compare(max_class[0])==0) //got a hit
			{
				confusion_matrix[max_class[0]][classes_[j_]]++;
				break;
			}
		}
		if(j_==classes_.size()) //no hit was found, just use any class
			confusion_matrix[max_class[0]][classes_[0]]++;
		
//		cvtColor(copy, copy, CV_HSV2BGR);
//		cvtColor(seg, seg, CV_HSV2BGR);
//		addWeighted(seg, 0.2, copy, 0.8, 1.0, seg);
//		imshow("seg", seg);
		
		Mat out; __img.copyTo(out);
		putText(out, max_class[0] + "!", Point(out.cols/2-100,out.rows/2-30), CV_FONT_HERSHEY_PLAIN, 3.0, Scalar(255), 2);
		if(max_class.size()>1) {
			putText(out, max_class[1] + "?", Point(out.cols/2-100,out.rows/2+30), CV_FONT_HERSHEY_PLAIN, 3.0, Scalar(255), 2);
		}
		imshow("out",out);
		waitKey(0);
		imwrite("output/"+filepath, out);
    }
	
	for(map<string,map<string,int> >::iterator it = confusion_matrix.begin(); it != confusion_matrix.end(); ++it) {
		cout << (*it).first << " -> ";
		for(map<string,int>::iterator it1 = (*it).second.begin(); it1 != (*it).second.end(); ++it1) {
			cout << (*it1).first << ":" << (*it1).second << endl;
		}
//		cout << endl;
	}
	
	cout << endl;

}