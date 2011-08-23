/*
 *  manual_classifier.cpp
 *  FoodcamClassifier
 *
 *  Created by Roy Shilkrot on 8/19/11.
 *  Copyright 2011 MIT. All rights reserved.
 *
 */


#include <iostream>
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

Point origin;
Rect selection;
Mat image;
bool selectObject;

void onMouse( int event, int x, int y, int, void* )
{
    if( selectObject )
    {
        selection.x = MIN(x, origin.x);
        selection.y = MIN(y, origin.y);
        selection.width = std::abs(x - origin.x);
        selection.height = std::abs(y - origin.y);
		
        selection &= Rect(0, 0, image.cols, image.rows);
    }
	
    switch( event )
    {
		case CV_EVENT_LBUTTONDOWN:
			origin = Point(x,y);
			selection = Rect(x,y,0,0);
			selectObject = true;
			break;
		case CV_EVENT_LBUTTONUP:
			selectObject = false;
			break;
    }
}

string char_to_class(char c) {
		switch (c) {
			case 'h':
			case 'H':
				return "chinese";
				break;
			case 'p':
			case 'P':
				return "pizza";
				break;
			case 'i':
			case 'I':
				return "indian";
				break;
			case 'w':
			case 'W':
				return "wraps";
				break;
			case 's':
			case 'S':
				return "sandwiches";
				break;
			case 'a':
			case 'A':
				return "salad";
				break;
			case 'c':
			case 'C':
				return "cookies";
				break;
			case 'm':
			case 'M':
				return "mexican";
				break;
			case 'f':
			case 'F':
				return "fruit_veggie";
				break;
			case 'l':
			case 'L':
				return "misc";
				break;
			case 't':
			case 'T':
				return "italian";
				break;
			default:
				return "misc";
		}
}

int main(int argc, char * const argv[]) {

	if(argc < 3) {
		cerr << "USAGE: manual_classifier <input_directory/> <output_file.txt>"<<endl;
		return 1;
	}
	string dir(argv[1]), filepath;
	DIR *dp;
	struct dirent *dirp;
	struct stat filestat;
	
	cout << "C'h'inease\nI't'alian\n'P'izza\n'I'ndian\n'W'raps\n'S'andwiches\nS'a'lad\n'C'ookies/Cake\n'M'exican\n'F'ruit/Veggies\nMisce'l'laneous" << endl;
	
	vector<pair<string, string> > classified;

	namedWindow("pic");
	setMouseCallback( "pic", onMouse, 0 );
	
	ifstream ifs(argv[2],ifstream::in);
	
	set<string> files_already_listed;
	if (ifs.is_open() && !ifs.eof()) {
		//something in here, get everything already listed
		char buf[255];
		while (!ifs.eof()) {
			ifs.getline(buf, 255);
			string line(buf);
			files_already_listed.insert(line.substr(0, line.find(" ")));
		}
	}
	ifs.close();
	
	ofstream ofs(argv[2], fstream::app);
	
	int count = 0;
	Mat img;
	bool running = true;
	dp = opendir( dir.c_str() );
	while (count++ < 1000 && (dirp = readdir( dp )) && running)
    {
		filepath = dir + "/" + dirp->d_name;
		
		// If the file is a directory (or is in some way invalid) we'll skip it 
		if (stat( filepath.c_str(), &filestat )) continue; //can't be opened...
		if (S_ISDIR( filestat.st_mode ))         continue; //a directory
		if (dirp->d_name[0] == '.')					 continue; //hidden file!
		if (files_already_listed.count(filepath)>0) continue; //already did that one
		
		ofs << filepath;
		img = imread(filepath);
		Point text_place(20,40);
		while (true) {
			img.copyTo(image);
			if( selection.width > 0 && selection.height > 0 )
			{
				Mat roi(image, selection);
				bitwise_not(roi, roi);
			}
			imshow("pic", image);
			int c = waitKey(10);
			
			if (c == ' ') {
				ofs << endl;
				break;
			} else if (c == 27) {
				running = false;
				break;
			} else if (c != -1) {
				if(selection.width != 0)
					ofs << " " << selection.x << "," << selection.y << "," << selection.width << "," << selection.height;
				ofs << " " << char_to_class(c);
				putText(img, char_to_class(c), text_place, CV_FONT_HERSHEY_PLAIN, 3.0, Scalar(255), 2);
				text_place += Point(0,40);
				selection = Rect();
			}
		}
    }
	closedir(dp);
	
	
	for (int i=0; i<classified.size(); i++) {
		ofs << classified[i].first << " " << classified[i].second << endl;
	}
	ofs.close();
}