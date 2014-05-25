/*
 * initParams.cpp
 *
 *  Created on: 2014-5-4
 *      Author: peng
 */

#include "initParams.h"
#include <string>

template<typename T>
void GetParameter(T &data, string query_str, ptree pt) {
	data = pt.get(query_str, data);
}

void PropertyTree2Params(ptree *pt, Params *params) {
	// Debug Part
	debugPart *debug = &params->debug;
	ptree pt_debug = pt->get_child("debug");
	GetParameter(debug->isDebug, "isDebug", pt_debug);
	GetParameter(debug->debugSize, "debugSize", pt_debug);
	GetParameter(debug->isRestart, "isRestart", pt_debug);
	// Data Part
	dataPart *data = &params->data;
	ptree pt_data = pt->get_child("data");
	GetParameter(data->imageSize.height, "imageSize.height", pt_data);
	GetParameter(data->imageSize.width, "imageSize.width", pt_data);
	GetParameter(data->WindowSize, "WindowSize", pt_data);
	GetParameter(data->StrideSize, "StrideSize", pt_data);
	GetParameter(data->ChannelSize, "ChannelSize", pt_data);
	GetParameter(data->dataSize_per_img.height, "dataSize_per_img.height",
			pt_data);
	GetParameter(data->dataSize_per_img.width, "dataSize_per_img.width",
			pt_data);
	GetParameter(data->data_per_img, "data_per_img", pt_data);
	GetParameter(data->data_D, "data_D", pt_data);
	GetParameter(data->data_reduceD, "data_reduceD", pt_data);
	// Path Part
	pathPart *path = &params->path;
	ptree pt_path = pt->get_child("path");
	GetParameter(path->isLinux, "isLinux", pt_path);
	GetParameter(path->dataFloder, "dataFloder", pt_path);
	GetParameter(path->trainFloder, "trainFloder", pt_path);
	GetParameter(path->validFloder, "validFloder", pt_path);
	GetParameter(path->testFloder, "testFloder", pt_path);
	GetParameter(path->satFloder, "satFloder", pt_path);
	GetParameter(path->mapFloder, "mapFloder", pt_path);
	GetParameter(path->cacheFloder, "cacheFloder", pt_path);
	GetParameter(path->cacheRBM, "cacheRBM", pt_path);
	GetParameter(path->cacheEpochRBM, "cacheEpochRBM", pt_path);
	GetParameter(path->cacheNN, "cacheNN", pt_path);
	GetParameter(path->cacheEpochNN, "cacheEpochNN", pt_path);
	GetParameter(path->cacheTestY, "cacheTestY", pt_path);
	GetParameter(path->cacheTrainY, "cacheTrainY", pt_path);
	GetParameter(path->cacheImageNum, "cacheImageNum", pt_path);

}
void Params2PropertyTree(Params *params, ptree *pt) {
	// Debug Part
	ptree pt_debug;
	debugPart *debug = &params->debug;
	pt_debug.put("isDebug", debug->isDebug);
	pt_debug.put("debugSize", debug->debugSize);
	pt_debug.put("isRestart", debug->isRestart);
	pt->put_child("debug", pt_debug);
	// Data Part
	ptree pt_data;
	dataPart *data = &params->data;
	pt_data.put("imageSize.height", data->imageSize.height);
	pt_data.put("imageSize.width", data->imageSize.width);
	pt_data.put("WindowSize", data->WindowSize);
	pt_data.put("StrideSize", data->StrideSize);
	pt_data.put("ChannelSize", data->ChannelSize);
	pt_data.put("dataSize_per_img.height", data->dataSize_per_img.height);
	pt_data.put("dataSize_per_img.width", data->dataSize_per_img.width);
	pt_data.put("data_per_img", data->data_per_img);
	pt_data.put("data_D", data->data_D);
	pt_data.put("data_reduceD", data->data_reduceD);
	pt->put_child("data", pt_data);
	// Path Part
	ptree pt_path;
	pathPart *path = &params->path;
	pt_path.put("isLinux", path->isLinux);
	pt_path.put("dataFloder", path->dataFloder);
	pt_path.put("trainFloder", path->trainFloder);
	pt_path.put("validFloder", path->validFloder);
	pt_path.put("testFloder", path->testFloder);
	pt_path.put("satFloder", path->satFloder);
	pt_path.put("mapFloder", path->mapFloder);
	pt_path.put("cacheFloder", path->cacheFloder);
	pt_path.put("cacheRBM", path->cacheRBM);
	pt_path.put("cacheEpochRBM", path->cacheEpochRBM);
	pt_path.put("cacheNN", path->cacheNN);
	pt_path.put("cacheEpochNN", path->cacheEpochNN);
	pt_path.put("cacheTestY", path->cacheTestY);
	pt_path.put("cacheTrainY", path->cacheTrainY);
	pt_path.put("cacheImageNum", path->cacheImageNum);
	pt->put_child("path", pt_path);
}

/*
 * This is the function to initialize the configuration params in the project.
 */
Params* initParams(string path) {
	Params* params = new Params();
	ptree *pt = new ptree();
	if (exists(path)) {
		read_json(path, *pt);
		PropertyTree2Params(pt, params);
	} else {
		Params2PropertyTree(params, pt);
		write_json(path, *pt);
	}

	delete pt;
	return params;
}

//int main(int argc, char **argv) {
//	Params *params = initParams();
//	cout<<stoi("1");
//	delete params;
//	return 0;
//}

// eof

