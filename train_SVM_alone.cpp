/*
 *  train_SVM_alone.cpp
 *  
 *
 *  Created by Roy Shilkrot on 8/22/11.
 *  Copyright 2011 MIT. All rights reserved.
 *
 */
#include "train_bovw.h"

int main() {
	cout << "load from file.."<<endl;
	map<string,Mat> classes_training_data;
	FileStorage fs("training_samples.yml",FileStorage::READ);
	vector<string> classes_names;
	fs["classes"] >> classes_names;
	for (vector<string>::iterator it = classes_names.begin(); it != classes_names.end(); ++it) {
		fs[(*it)] >> classes_training_data[*it];
	}
	
	cout << "train SVM.." <<endl;
	string file_postfix = "with_colors";
	Mat& one_class = (*(classes_training_data.begin())).second;
	trainSVM(classes_training_data, file_postfix, one_class.cols, one_class.type());
}