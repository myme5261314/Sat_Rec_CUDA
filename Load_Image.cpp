/*
 * Load_Image.cpp
 *
 *  Created on: 2014-5-7
 *      Author: peng
 */

#include "Load_Image.h"

void dispVector(vector<string> vec) {
	BOOST_FOREACH(string i, vec){
	std::cout << i << std::endl;
}
}

bool strStartWith(string s, string s_sub) {
	return s.find(s_sub) == 0;
}
bool strEndWith(string s, string s_sub) {
	return s.rfind(s_sub) == s.length() - s_sub.length();
}

vector<string> excludeFileList(vector<string> raw_filelist,
		string excludeKeyWord, bool isBlackList) {
	std::vector<string> new_filelist;
	for (std::vector<string>::iterator i = raw_filelist.begin();
			i != raw_filelist.end(); ++i) {
		if (strEndWith(*i, excludeKeyWord) == !isBlackList) {
			new_filelist.push_back(*i);
		}
	}
	std::sort(new_filelist.begin(), new_filelist.end());
	return new_filelist;
}

vector<string> getDirList(string s_path) {
	vector<string> filelist;
	assert(exists(s_path));
	boost::filesystem::path targetDir(s_path);
	boost::filesystem::directory_iterator it(targetDir), eod;
	BOOST_FOREACH(boost::filesystem::path const &p, std::make_pair(it, eod)){
	if(is_regular_file(p))
	{
		filelist.push_back(p.string());
	}
}
	return filelist;
}

std::vector<string> getDirList(boost::filesystem::path p) {
	return getDirList(p.string());
}

std::vector<string> extractFileList(vector<string> raw_filelist,
		size_t extractNum) {
	vector<string> new_filelist;
	if (extractNum==0) extractNum = raw_filelist.size();
	for (size_t i = 0; i < extractNum; ++i) {
		new_filelist.push_back(raw_filelist[i]);
	}
	return new_filelist;
}

vector<cv::Mat> batchLoadImage(vector<string> v_path) {
	vector<cv::Mat> ImgVec;
	for (std::vector<string>::const_iterator i = v_path.begin();
			i != v_path.end(); ++i) {
		Mat image;
		image = imread(*i, CV_LOAD_IMAGE_COLOR);   // Read the file
		if (!image.data) {
			cout << "Could not open or find the image" << std::endl;
		} else {
			ImgVec.push_back(image);
		}
	}
	return ImgVec;
}

void testLoad_Image(Params *params) {
	boost::filesystem::path root(params->path.dataFloder);
	boost::filesystem::path trainSat = root / params->path.trainFloder
			/ params->path.satFloder;
	std::vector<string> trainSat_vec = extractFileList(
			excludeFileList(getDirList(trainSat), ".tiff", false),
			params->debug.debugSize);
	dispVector(trainSat_vec);
	boost::filesystem::path trainMap = root / params->path.trainFloder
			/ params->path.mapFloder;
	std::vector<string> trainMap_vec = extractFileList(
			excludeFileList(getDirList(trainMap), ".tif", false),
			params->debug.debugSize);
	dispVector(trainMap_vec);
	std::vector<cv::Mat> trainMapImgVec = batchLoadImage(trainMap_vec);
	std::vector<cv::Mat> trainSatImgVec = batchLoadImage(trainSat_vec);
//	namedWindow( "Display window", WINDOW_AUTOSIZE );// Create a window for display.
//	imshow( "Display window", trainMapImgVec[0] );                   // Show our image inside it.
//	namedWindow( "Sat Display", WINDOW_AUTOSIZE);
//	imshow( "Sat Display", trainSatImgVec[0]);
//	waitKey(0);                                          // Wait for a keystroke in the window
	std::cout << cv::gpu::getCudaEnabledDeviceCount() << std::endl;
}

