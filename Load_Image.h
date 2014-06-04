/*
 * Load_Image.h
 *
 *  Created on: 2014-5-7
 *      Author: peng
 */

#ifndef LOAD_IMAGE_H_
#define LOAD_IMAGE_H_

#include <iostream>
#include <algorithm>
#include <vector>
#include <string>
#include <boost/foreach.hpp>
#include <boost/filesystem.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/gpu/gpu.hpp>
#include "Params.h"

using namespace std;

using namespace boost::filesystem;
using namespace cv;

void dispVector(vector<string> vec);
vector<string> getDirList(string s_path);
vector<string> getDirList(boost::filesystem::path p);
std::vector<string> extractFileList(vector<string> raw_filelist,
		size_t extractNum);
vector<string> excludeFileList(vector<string> raw_filelist,
		string excludeKeyWord, bool isBlackList);
vector<cv::Mat> batchLoadImage(vector<string> v_path);
void testLoad_Image(Params *params);

#endif /* LOAD_IMAGE_H_ */
