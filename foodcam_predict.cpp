/*
 *  foodcam_predict.cpp
 *  FoodcamClassifier
 *
 *  Created by Roy Shilkrot on 8/23/11.
 *  Copyright 2011 MIT. All rights reserved.
 *
 */

#include "predict_common.h"

int main(int argc, char* argv[]) {
	if (argc < 2) {
		cerr << "USAGE: ./foodcam-predict <image_file.png>" << endl;
		return 1;
	}
	
	FoodcamPredictor predictor;
	predictor.setDebug(false);
	
	Mat __img = imread(argv[1]),_img;
	if(__img.size() != Size(640,480)) {
		cerr << "Foodcam images are 640x480, you provided " << Point(__img.size()) << endl;
		return 1;
	}
	
	vector<string> max_class;
	predictor.evaluateOneImage(__img,max_class);	
	
	cout << max_class[0];
	if (max_class.size()>1) {
		cout << "," << max_class[1];
	}
	cout << endl;
	
	return 0;
}