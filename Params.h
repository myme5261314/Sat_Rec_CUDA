/*
 * Params.h
 *
 *  Created on: 2014-5-4
 *      Author: peng
 */

#ifndef PARAMS_H_
#define PARAMS_H_

#ifdef WIN32
static bool os_linux = false;
#else
static bool os_linux = true;
#endif

#include <assert.h>
#include <math.h>
#include <string>
#include <opencv2/opencv.hpp>
using namespace std;

typedef struct imgSize {
	size_t height;
	size_t width;
} matSize;

size_t matSize2Size(matSize mat);
matSize imgSize2DataMatSize(imgSize ims, size_t WindowSize, size_t StrideSize);

struct debugPart {
	bool isDebug = 1;
	size_t debugSize = 10;
	bool isRestart = 0;
};

struct dataPart {
	imgSize imageSize = { 1500, 1500 };
	size_t WindowSize = 64;
	size_t StrideSize = 16;
	size_t ChannelSize = 3;
	matSize dataSize_per_img = imgSize2DataMatSize(this->imageSize,
			this->WindowSize, this->StrideSize);
	size_t data_per_img = matSize2Size(this->dataSize_per_img);
	size_t data_D = this->WindowSize * this->WindowSize * this->ChannelSize;
	size_t data_reduceD = this->WindowSize * this->WindowSize;

	cv::Mat premean;
	cv::Mat prestd;
	cv::Mat Ureduce;
	cv::Mat postmean;
	cv::Mat poststd;
};

struct pathPart {
	// OS
	bool isLinux = os_linux;
	// Floder Path
	string dataFloder =
			this->isLinux ?
					"~/Sat_Rec_Dataset/Mass_Roads/" :
					"E:/Sat_Rec_Dataset/Mass_Roads/";
	string trainFloder = "Train/";
	string validFloder = "Train/";
	string testFloder = "Test/";
	string satFloder = "Sat/";
	string mapFloder = "Map/";
	string cacheFloder = "cache/";
	// Cache Path
	string cachePreMean = "premean.dat";
	string cachePreStd = "prestd.dat";
	string cachePca = "pca.dat";
	string cachePostMean = "postmean.dat";
	string cachePostStd = "poststd.dat";
	string cacheRBM = "rbm.dat";
	string cacheEpochRBM = "epochrbm.dat";
	string cacheNN = "nn.dat";
	string cacheEpochNN = "epochnn.dat";
	string cacheTestY = "testy.dat";
	string cacheTrainY = "trainy.dat";
	size_t cacheImageNum = 5;
};

struct Params {
	// Debug Part
	debugPart debug;
	// Data Part
	dataPart data;
	// Path Part
	pathPart path;
};

#endif /* PARAMS_H_ */
