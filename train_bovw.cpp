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
	
	if (argv.size() < 3) {
		cout << "USAGE: train_bovw <vocabulary_file.yml> <postfix_for_output>"<<endl;
		return 1;
	}
	
	cout << "-------- train BOVW SVMs -----------" << endl;
	cout << "read vocabulary form file"<<endl;
	Mat vocabulary;
	FileStorage fs(argv[1], FileStorage::READ);
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
	
	//setup training data for classifiers
	map<string,Mat> classes_training_data; classes_training_data.clear();
	
	cout << "extract_training_samples.." << endl;
	extract_training_samples(detector, bowide, classes_training_data);
	
	cout << "got " << classes_training_data.size() << " classes." <<endl;
	for (map<string,Mat>::iterator it = classes_training_data.begin(); it != classes_training_data.end(); ++it) {
		cout << " class " << (*it).first << " has " << (*it).second.rows << " samples"<<endl;
	}
	
	cout << "train SVMs\n";
	string postfix = argv[2];
	trainSVM(classes_training_data, postfix, bowide.descriptorSize(), bowide.descriptorType());

	return 0;
}
