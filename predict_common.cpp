/*
*  predict_common.cpp
*  FoodcamClassifier
*
*  Created by Roy Shilkrot on 8/23/11.
*  Copyright 2011 MIT. All rights reserved.
*
*/

#include "predict_common.h"

FoodcamPredictor::FoodcamPredictor() {
	debug = false;
	initSVMs();
	initColors();
	initVocabulary();
	Ptr<FeatureDetector > _detector(new SurfFeatureDetector());
	Ptr<DescriptorMatcher > _matcher(new BruteForceMatcher<L2<float> >());
	Ptr<DescriptorExtractor > _extractor(new OpponentColorDescriptorExtractor(Ptr<DescriptorExtractor>(new SurfDescriptorExtractor())));
	matcher = _matcher;
	detector = _detector;
	extractor = _extractor;
	bowide = Ptr<BOWImgDescriptorExtractor>(new BOWImgDescriptorExtractor(extractor,matcher));
	bowide->setVocabulary(vocabulary);
	background = imread("background.png");
}

void FoodcamPredictor::initColors() {
	int ccount = 0;
	for (map<string,CvSVM>::iterator it = classes_classifiers.begin(); it != classes_classifiers.end(); ++it) {
		classes_colors[(*it).first] = Scalar((float)(ccount++)/(float)(classes_classifiers.size())*180.0f,255,255);
		if(debug) cout << "class " << (*it).first << " color " << classes_colors[(*it).first].val[0] << endl;
	}
}	

void FoodcamPredictor::initSVMs() {
	string dir, filepath;
	DIR *dp;
	struct dirent *dirp;
	struct stat filestat;
	
	if(debug) cout << "load SVM classifiers" << endl;
	dir = ".";
	dp = opendir( dir.c_str() );
	
	while ((dirp = readdir( dp )))
    {
		filepath = dir + "/" + dirp->d_name;
		
		// If the file is a directory (or is in some way invalid) we'll skip it 
		if (stat( filepath.c_str(), &filestat )) continue;
		if (S_ISDIR( filestat.st_mode ))         continue;
		if (filepath.find("SVM_classifier_with_color") != string::npos)
		{
			string class_ = filepath.substr(filepath.rfind('_')+1,filepath.rfind('.')-filepath.rfind('_')-1);
			if (debug) cout << "load " << filepath << ", class: " << class_ << endl;
			classes_classifiers.insert(pair<string,CvSVM>(class_,CvSVM()));
			classes_classifiers[class_].load(filepath.c_str());
		}
	}
	closedir(dp);
}

void FoodcamPredictor::initVocabulary() {
	if (debug) cout << "read vocabulary form file"<<endl;
	FileStorage fs("vocabulary_color_1000.yml", FileStorage::READ);
	fs["vocabulary"] >> vocabulary;
	fs.release();	
}	

void FoodcamPredictor::evaluateOneImage(Mat& __img, vector<string>& out_classes) {
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
	
	Rect crop_rect(0,100,640,480-100);
	__img = __img(crop_rect);		//crop off top section
	fg_mask = fg_mask(crop_rect);
	
	//_img.create(__img.size(), __img.type());
	//		cvtColor(__img, _img, CV_BGR2GRAY);
	//		equalizeHist(__img, _img);
	Mat copy; cvtColor(__img, copy, CV_BGR2HSV);
	
	vector<Point> check_points;
	//Sliding window approach.. (creating a vector here to ease the OMP parallel for-loop)
	int winsize = 200;
	map<string,pair<int,float> > found_classes;
	for (int x=0; x<__img.cols; x+=winsize/4) {
		for (int y=0; y<__img.rows; y+=winsize/4) {
			if (fg_mask.at<uchar>(y,x) == 0) {
				continue;
			}
			check_points.push_back(Point(x,y));
		}
	}
	
	if (debug) cout << "to check: " << check_points.size() << " points"<<endl;
	
	Mat seg = Mat::zeros(copy.size(),CV_8UC3);
	
#pragma omp parallel for
	for (int i = 0; i < check_points.size(); i++) {
		int x = check_points[i].x;
		int y = check_points[i].y;
		//			if (debug) cout << omp_get_thread_num() << " scan " << check_points[i] << endl;
		Mat img,response_hist;
		__img(Rect(x-winsize/2,y-winsize/2,winsize,winsize)&Rect(0,0,__img.cols,__img.rows)).copyTo(img);
		
		vector<KeyPoint> keypoints;
		detector->detect(img,keypoints);
		//				vector<vector<int> > pointIdxsOfClusters;
		bowide->compute(img, keypoints, response_hist); //, &pointIdxsOfClusters);
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
			//				if (debug) cout << "best class: " << minclass << " ("<<minf<<")"<<endl;
			if (debug) cout << "."; if (debug) cout.flush();
			//circle(copy, Point(x,y), 5, classes_colors[minclass], CV_FILLED);
			float dim = MAX(MIN(minf - 0.8f,0.3f),0.0f) / 0.3f; //dimming the color: [0.8,1.1] -> [0.0,1.0]
			Scalar color_(classes_colors[minclass].val[0], classes_colors[minclass].val[1], classes_colors[minclass].val[2] * dim); 
			
#pragma omp critical
			{
				putText(copy, minclass.substr(0, 4), Point(x-35,y+10), CV_FONT_HERSHEY_PLAIN, 2.0, Scalar(0,0,255), 2);
				circle(seg, check_points[i], winsize/5, color_, CV_FILLED);
				found_classes[minclass].first++;
				found_classes[minclass].second += minf;
			}
		}
		catch (cv::Exception) {
			continue;
		}
	}
	
	if (debug) cout << endl << "found classes: ";
	float max_class_f = FLT_MIN, max_class_f1 = FLT_MIN; string max_class, max_class1;
	vector<float> scores;
	for (map<string,pair<int,float> >::iterator it=found_classes.begin(); it != found_classes.end(); ++it) {
		float score = sqrtf((float)((*it).second.first) * (*it).second.second);
		if (score > 1e+10) {
			continue;	//an impossible score
		}
		scores.push_back(score);
		if (debug) cout << (*it).first << "(" << score << "),"; //<< (*it).second.first << "," << (*it).second.second / (float)(*it).second.first << "), ";
		if(score > max_class_f) { //1st place thrown off
			max_class_f1 = max_class_f;
			max_class1 = max_class;
			
			max_class_f = score;
			max_class = (*it).first;
		} else if (score >  max_class_f1) {	//2nd place thrown off
			max_class_f1 = score;
			max_class1 = (*it).first;
		}
	}
	if (debug) cout << endl;

	normalizeClassname(max_class);
	normalizeClassname(max_class1);

	Scalar mean_,stddev_;
//	meanStdDev(Mat(scores), mean_, stddev_);
	out_classes.clear();
	out_classes.push_back(max_class);
	if(max_class_f - max_class_f1 < 10) {
		//Forget about it: variance is low (~10), so result is undecicive, we should take both max-classes.
		out_classes.push_back(max_class1);
	}	
	
	if (debug) cout << "chosen class: " << max_class << ", (" << max_class1 << "?)" << endl;
}