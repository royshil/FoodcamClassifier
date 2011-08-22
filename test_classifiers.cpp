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
		if (filepath.find("SVM_classifier") != string::npos)
		{
			string class_ = filepath.substr(filepath.rfind('_')+1,filepath.rfind('.')-filepath.rfind('_'));
			cout << "load " << filepath << ", class: " << class_ << endl;
			classes_classifiers.insert(pair<string,CvSVM>(class_,CvSVM()));
			classes_classifiers[class_].load(filepath.c_str());
		}
	}
	
	cout << "read vocabulary form file"<<endl;
	Mat vocabulary;
	FileStorage fs("vocabulary_1000.yml", FileStorage::READ);
	fs["vocabulary"] >> vocabulary;
	fs.release();	
	
	Ptr<SurfFeatureDetector > detector(new SurfFeatureDetector()); //detector
	Ptr<DescriptorExtractor > extractor(new SurfDescriptorExtractor());//  extractor;
	Ptr<DescriptorMatcher > matcher(new BruteForceMatcher<L2<float> >());
	BOWImgDescriptorExtractor bowide(extractor,matcher);
	bowide.setVocabulary(vocabulary);
	
	cout << "------- test ---------\n";
	
	//evaluate
	dir = "foodcamimages/TEST";
	dp = opendir( dir.c_str() );
	count = 0;
	Mat img,response_hist;
	vector<KeyPoint> keypoints;
	
	map<string,Scalar> classes_colors;
	int ccount = 0;
	for (map<string,CvSVM>::iterator it = classes_classifiers.begin(); it != classes_classifiers.end(); ++it) {
		classes_colors[(*it).first] = Scalar((float)(ccount++)/(float)(classes_classifiers.size())*180.0f,255,255);
		cout << "class " << (*it).first << " color " << classes_colors[(*it).first].val[0] << endl;
	}
	
	Mat background = imread("background.png");
	while ((dirp = readdir( dp )))
    {
		count++;
		if(count > 65) break;
		if(count < 25) continue;
		
		filepath = dir + "/" + dirp->d_name;
		
		// If the file is a directory (or is in some way invalid) we'll skip it 
		if (stat( filepath.c_str(), &filestat )) continue;
		if (S_ISDIR( filestat.st_mode ))         continue;
		if (dirp->d_name[0] == '.')					 continue; //hidden file!
		
		cout << "eval file " << filepath << endl;
		
		Mat __img = imread(filepath),_img;
		
		Mat diff = (__img - background), diff_8UC1;
		
		cvtColor(diff, diff_8UC1, CV_BGR2GRAY);
//		imshow("img no back", diff_8UC1);
		Mat fg_mask = (diff_8UC1 > 5);
		GaussianBlur(fg_mask, fg_mask, Size(11,11), 5.0);
		fg_mask = fg_mask > 125;
		
//		{
//			Mat _out; __img.copyTo(_out, fg_mask);
//			imshow("foregroung", _out);
//			waitKey(0);
//		}
		
		__img = __img(Rect(100,100,640-200,480-100));//crop off top section
		//_img.create(__img.size(), __img.type());
		cvtColor(__img, _img, CV_BGR2GRAY);
		equalizeHist(_img, _img);
		Mat copy; cvtColor(__img, copy, CV_BGR2HSV);

		
		//Sliding window approach..
		int winsize = 300;
		set<string> found_classes;
		for (int x=0; x<_img.cols; x+=winsize/6) {
			for (int y=0; y<_img.rows; y+=winsize/6) {
				if (fg_mask.at<uchar>(y+100, x+100) == 0) {
					continue;
				}
				_img(Rect(x-winsize/2,y-winsize/2,winsize,winsize)&Rect(0,0,_img.cols,_img.rows)).copyTo(img);
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
//						if ((*it).first == "misc.") {
//							//						cout << "misc = " << res << endl;
//							continue;
//						}
						if (res < minf) {
							minf = res;
							minclass = (*it).first;
						}
					}
	//				cout << "best class: " << minclass << " ("<<minf<<")"<<endl;
					cout << ".";
					//circle(copy, Point(x,y), 5, classes_colors[minclass], CV_FILLED);
					float dim = 1.0f - MAX(MIN(minf - 0.8f,0.3f),0.0f) / 0.3f; //dimming the color: [0.8,1.1] -> [0.0,1.0]
					Scalar color_(classes_colors[minclass].val[0], classes_colors[minclass].val[1], classes_colors[minclass].val[2] * dim); 
					putText(copy, minclass.substr(0, 2), Point(x,y), CV_FONT_HERSHEY_PLAIN, 2.0, color_, 2);
					found_classes.insert(minclass);
				}
				catch (cv::Exception) {
					continue;
				}
			}
		}
		cout << "found classes: ";
		for (set<string>::iterator it=found_classes.begin(); it != found_classes.end(); ++it) {
			cout << (*it) << ", ";
		}
		cout << endl;
		cout << endl;
		cvtColor(copy, copy, CV_HSV2BGR);
		imshow("pic", copy);
		 waitKey(0);
		//TODO: sliding window on the image to crop smaller secions and classify each section (segmentation? booya)
		
		//		cout << ".";
    }
	cout << endl;
	closedir( dp );

}