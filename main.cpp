#include <stdio.h>
#include <stdlib.h>

#include <opencv2/opencv.hpp>
#include <fstream>
#include <iostream>
#include <string>

#include <dirent.h>
#include <unistd.h>
#include <sys/stat.h>
#include <sys/types.h>

using namespace cv;
using namespace std;

int main (int argc, char * const argv[]) {
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
	int count = 0;
	while (count++ < 15 && (dirp = readdir( dp )))
    {
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
    
	BOWKMeansTrainer bowtrainer(150); //num clusters
	bowtrainer.add(training_descriptors);
	cout << "cluster BOW features" << endl;
	Mat vocabulary = bowtrainer.cluster();
	
	Ptr<DescriptorMatcher > matcher(new BruteForceMatcher<L2<float> >());
	BOWImgDescriptorExtractor bowide(extractor,matcher);
	bowide.setVocabulary(vocabulary);
	
	//setup training data for classifiers
	map<string,Mat> classes_training_data; classes_training_data.clear();
	
	cout << "------- train SVMs ---------\n";

	Mat response_hist;
	cout << "look in train data"<<endl;
	count = 0;
	char buf[255];
	ifstream ifs("training.txt");
	int total_samples = 0;
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
		cout << ".";
//		putText(img, c_, Point(20,20), CV_FONT_HERSHEY_PLAIN, 2.0, Scalar(255), 2);
//		imshow("pic",img);
		bowide.compute(img, keypoints, response_hist);
		
		if(classes_training_data.count(c_) == 0) { //not yet created...
			classes_training_data[c_].create(0,response_hist.cols,response_hist.type());
		}
		classes_training_data[c_].push_back(response_hist);
		total_samples++;
//		waitKey(0);
	} while (!ifs.eof());
	cout << endl;
	
	//train 1-vs-all SVMs
	map<string,CvSVM> classes_classifiers;
	for (map<string,Mat>::iterator it = classes_training_data.begin(); it != classes_training_data.end(); ++it) {
		string class_ = (*it).first;
		cout << "training class: " << class_ << ".." << endl;
		
		Mat samples(0,response_hist.cols,response_hist.type());
		Mat labels(0,1,CV_32FC1);
		
		//copy class samples and label
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
		classes_classifiers[class_].train(samples_32f,labels);
	}
	
	cout << "------- test ---------\n";

	//evaluate
	dir = "/Users/royshilkrot/Downloads/foodcamimages/TEST";
	dp = opendir( dir.c_str() );
	count = 0;
	while (count++ < 5 && (dirp = readdir( dp )))
    {
		filepath = dir + "/" + dirp->d_name;
		
		// If the file is a directory (or is in some way invalid) we'll skip it 
		if (stat( filepath.c_str(), &filestat )) continue;
		if (S_ISDIR( filestat.st_mode ))         continue;
		if (dirp->d_name[0] == '.')					 continue; //hidden file!
		
		cout << "eval file " << filepath << endl;
		
		img = imread(filepath);
		bowide.compute(img, keypoints, response_hist);
		
		//test vs. SVMs
		for (map<string,CvSVM>::iterator it = classes_classifiers.begin(); it != classes_classifiers.end(); ++it) {
			float res = (*it).second.predict(response_hist,false);
			cout << "class: " << (*it).first << ", response: " << res << endl;
		}
		
//		cout << ".";
    }
	cout << endl;
	closedir( dp );
	
	
	cout <<"done"<<endl;
    return 0;
}
