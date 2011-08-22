/*
 
 *  make_test_background_image.cpp
 *  FoodcamClassifier
 *
 *  Created by Roy Shilkrot on 8/21/11.
 *  Copyright 2011 MIT. All rights reserved.
 *
 */

#include "make_test_background_image.h"

int main(int argc, char** argv) {
	string dir, filepath;
	DIR *dp;
	struct dirent *dirp;
	struct stat filestat;
	
	//get images
	dir = "foodcamimages/TEST";
	dp = opendir( dir.c_str() );
	int count = 0;
	Mat accum;
	while ((dirp = readdir( dp )))
    {
		count++;
		
		filepath = dir + "/" + dirp->d_name;
		
		// If the file is a directory (or is in some way invalid) we'll skip it 
		if (stat( filepath.c_str(), &filestat )) continue;
		if (S_ISDIR( filestat.st_mode ))         continue;
		if (dirp->d_name[0] == '.')					 continue; //hidden file!
		
		cout << "eval file " << filepath << endl;

		Mat img = imread(filepath),img64;
		img.convertTo(img64, CV_64FC3);
		
		if (!accum.data) {
			accum.create(img.size(), CV_64FC3);
		}
		if (img64.size() == accum.size()) {
			accum += img64;
		}
	}
	
	accum /= count;
	Mat accum_8UC3; accum.convertTo(accum_8UC3, CV_8UC3);
	
	imwrite("background.png", accum_8UC3);
	
	imshow("accum", accum_8UC3);
	waitKey(0);
}