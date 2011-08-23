/*
 *  test_classifiers.cpp
 *  FoodcamClassifier
 *
 *  Created by Roy Shilkrot on 8/20/11.
 *  Copyright 2011 MIT. All rights reserved.
 *
 */

#include "test_classifiers.h"

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

int main(int argc, char** argv) {
	string dir, filepath;
	DIR *dp;
	struct dirent *dirp;
	struct stat filestat;
	
	cout << "load SVM classifiers" << endl;
	dir = ".";
	dp = opendir( dir.c_str() );
	
	map<string,CvSVM> classes_classifiers;
	int count = 0;
	while ((dirp = readdir( dp )))
    {
		filepath = dir + "/" + dirp->d_name;
		
		// If the file is a directory (or is in some way invalid) we'll skip it 
		if (stat( filepath.c_str(), &filestat )) continue;
		if (S_ISDIR( filestat.st_mode ))         continue;
		if (filepath.find("SVM_classifier_with_color") != string::npos)
		{
			string class_ = filepath.substr(filepath.rfind('_')+1,filepath.rfind('.')-filepath.rfind('_')-1);
			cout << "load " << filepath << ", class: " << class_ << endl;
			classes_classifiers.insert(pair<string,CvSVM>(class_,CvSVM()));
			classes_classifiers[class_].load(filepath.c_str());
		}
	}
	
	cout << "read vocabulary form file"<<endl;
	Mat vocabulary;
	FileStorage fs("vocabulary_color_1000.yml", FileStorage::READ);
	fs["vocabulary"] >> vocabulary;
	fs.release();	
	
	Ptr<FeatureDetector > detector(new SurfFeatureDetector()); //detector
	//Ptr<DescriptorExtractor > extractor(new SurfDescriptorExtractor());//  extractor;
	Ptr<DescriptorExtractor > extractor(
		new OpponentColorDescriptorExtractor(
			 Ptr<DescriptorExtractor>(new SurfDescriptorExtractor())
			 )
		);
	Ptr<DescriptorMatcher > matcher(new BruteForceMatcher<L2<float> >());
	BOWImgDescriptorExtractor bowide(extractor,matcher);
	bowide.setVocabulary(vocabulary);
	
	cout << "------- test ---------\n";
	
	//evaluate
	dir = "foodcamimages/TEST";
	dp = opendir( dir.c_str() );
	count = 0;
//	Mat response_hist;
	
	
	map<string,Scalar> classes_colors;
	int ccount = 0;
	for (map<string,CvSVM>::iterator it = classes_classifiers.begin(); it != classes_classifiers.end(); ++it) {
		classes_colors[(*it).first] = Scalar((float)(ccount++)/(float)(classes_classifiers.size())*180.0f,255,255);
		cout << "class " << (*it).first << " color " << classes_colors[(*it).first].val[0] << endl;
	}
	
	ifstream ifs("test.txt",ifstream::in);
	char buf[255];
	vector<string> lines; 
	while(!ifs.eof()) {// && count++ < 30) {
		ifs.getline(buf, 255);
		lines.push_back(buf);
	}	
	ifs.close();
	cout << "total " << lines.size() << " samples to scan" <<endl;
	
	map<string,map<string,int> > confusion_matrix;
	for (map<string,CvSVM>::iterator it = classes_classifiers.begin(); it != classes_classifiers.end(); ++it) {
		for (map<string,CvSVM>::iterator it1 = classes_classifiers.begin(); it1 != classes_classifiers.end(); ++it1) {
			string class1 = ((*it).first.compare("cake")==0) ? "cookies" : (*it).first;
			string class2 = ((*it1).first.compare("cake")==0) ? "cookies" : (*it1).first;
			confusion_matrix[class1][class2] = 0;
		}
	}
	
	Mat background = imread("background.png");
//	while ((dirp = readdir( dp )))
	for (int i = 0; i < lines.size(); i++) {
//		count++;
//		if(count > 65) break;
//		if(count < 25) continue;
//		
//		filepath = dir + "/" + dirp->d_name;
//		
//		// If the file is a directory (or is in some way invalid) we'll skip it 
//		if (stat( filepath.c_str(), &filestat )) continue;
//		if (S_ISDIR( filestat.st_mode ))         continue;
//		if (dirp->d_name[0] == '.')					 continue; //hidden file!
		
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
		
		Mat diff = (__img - background), diff_8UC1;
		
		cvtColor(diff, diff_8UC1, CV_BGR2GRAY);
//		imshow("img no back", diff_8UC1);
		Mat fg_mask = (diff_8UC1 > 5);
		GaussianBlur(fg_mask, fg_mask, Size(11,11), 5.0);
		fg_mask = fg_mask > 50;

//		{
//			Mat _out; __img.copyTo(_out, fg_mask);
//			imshow("foregroung", _out);
//			imshow("to scan",__img);
//			waitKey(0);
//		}

		__img = __img(Rect(100,100,640-200,480-100));		//crop off top section
		fg_mask = fg_mask(Rect(100,100,640-200,480-100));
		
		//_img.create(__img.size(), __img.type());
//		cvtColor(__img, _img, CV_BGR2GRAY);
//		equalizeHist(__img, _img);
		Mat copy; cvtColor(__img, copy, CV_BGR2HSV);

		vector<Point> check_points;
		//Sliding window approach..
		int winsize = 300;
		map<string,pair<int,float> > found_classes;
		for (int x=0; x<__img.cols; x+=winsize/4) {
			for (int y=0; y<__img.rows; y+=winsize/4) {
				if (fg_mask.at<uchar>(y,x) == 0) {
					continue;
				}
				check_points.push_back(Point(x,y));
			}
		}
		
		cout << "to check: " << check_points.size() << " points"<<endl;
		
		#pragma omp parallel for
		for (int i = 0; i < check_points.size(); i++) {
			int x = check_points[i].x;
			int y = check_points[i].y;
//			cout << omp_get_thread_num() << " scan " << check_points[i] << endl;
			Mat img,response_hist;
			__img(Rect(x-winsize/2,y-winsize/2,winsize,winsize)&Rect(0,0,__img.cols,__img.rows)).copyTo(img);
			
			vector<KeyPoint> keypoints;
			detector->detect(img,keypoints);
//				vector<vector<int> > pointIdxsOfClusters;
			bowide.compute(img, keypoints, response_hist); //, &pointIdxsOfClusters);
			if (response_hist.cols == 0 || response_hist.rows == 0) {
				continue;
			}
			
	//		drawKeypoints(img, keypoints, img, Scalar(0,0,255));
	//		for (int i = 0; i < pointIdxsOfClusters.size(); i++) {
	//			if(pointIdxsOfClusters[i].size()>0) {
	//				Scalar clr(i/1000.0*255.0,0,0);
	//				for (int j = 0; j < pointIdxsOfClusters[i].size(); j++) {
	//					circle(img, keypoints[pointIdxsOfClusters[i][j]].pt, 1, clr, 2);
	//				}
	//			}
	//		}
	//		imshow("pic",img);
			
			//test vs. SVMs
			try {
				float minf = FLT_MAX; string minclass;
				for (map<string,CvSVM>::iterator it = classes_classifiers.begin(); it != classes_classifiers.end(); ++it) {
					float res = (*it).second.predict(response_hist,true);
					if ((*it).first == "misc" && res > 0.9) {
						continue;
					}
					if(res > 1.0) continue;
					if (res < minf) {
						minf = res;
						minclass = (*it).first;
					}
				}
//				cout << "best class: " << minclass << " ("<<minf<<")"<<endl;
				cout << "."; cout.flush();
				//circle(copy, Point(x,y), 5, classes_colors[minclass], CV_FILLED);
				float dim = 1.0f - MAX(MIN(minf - 0.8f,0.3f),0.0f) / 0.3f; //dimming the color: [0.8,1.1] -> [0.0,1.0]
				Scalar color_(classes_colors[minclass].val[0], classes_colors[minclass].val[1], classes_colors[minclass].val[2] * dim); 
				
				#pragma omp critical
				{
//					putText(copy, minclass.substr(0, 2), Point(x,y), CV_FONT_HERSHEY_PLAIN, 2.0, color_, 2);
					found_classes[minclass].first++;
					found_classes[minclass].second += minf;
				}
			}
			catch (cv::Exception) {
				continue;
			}
		}
		
		cout << endl << "found classes: ";
		int max_class_i = 0; string max_class;
		for (map<string,pair<int,float> >::iterator it=found_classes.begin(); it != found_classes.end(); ++it) {
			cout << (*it).first << "(" << (*it).second.first << "," << (*it).second.second / (float)(*it).second.first << "), ";
			if((*it).second.first > max_class_i) {
				max_class_i = (*it).second.first;
				max_class = (*it).first;
			}
		}
		cout << endl;
		cout << "chosen class: " << max_class << endl;
		cout << "manual class: "; for(int j_=0;j_<classes_.size();j_++) cout << classes_[j_] << ",";
		cout << endl;
		
		int j_=0;
		for(;j_<classes_.size();j_++) {
			if(max_class.compare("cake")==0) max_class = "cookies";
			if(classes_[j_].compare(max_class)==0) //got a hit
			{
				confusion_matrix[max_class][classes_[j_]]++;
				break;
			}
		}
		if(j_==classes_.size()) //no hit was found, just use any class
			confusion_matrix[max_class][classes_[0]]++;

//		cvtColor(copy, copy, CV_HSV2BGR);
//		imshow("pic", copy);
//		 waitKey(0);

		//TODO: sliding window on the image to crop smaller secions and classify each section (segmentation? booya)
		
		//		cout << ".";
    }
	
	for(map<string,map<string,int> >::iterator it = confusion_matrix.begin(); it != confusion_matrix.end(); ++it) {
		//cout << (*it).first << ": ";
		for(map<string,int>::iterator it1 = (*it).second.begin(); it1 != (*it).second.end(); ++it1) {
			cout << (*it1).second << ",";
		}
		cout << endl;
	}
	
	cout << endl;
	closedir( dp );

}