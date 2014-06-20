/*
 * Mass_Roads.cpp
 *
 *  Created on: 2014-5-7
 *      Author: peng
 */

#include "Mass_Roads.h"



using namespace cv;

int main() {
	// Initialize Params.
	culaStatus s = culaInitialize();
	dispCULAStatus(s);
	testMatMultiplication();
	Params *params = initParams();
	params->path.dataFloder =
			"/media/F6F6B6AFF6B66F8B/Sat_Rec_Dataset/Mass_Roads/";
	params->debug.debugSize = 2;
	boost::property_tree::ptree pt;
	Params2PropertyTree(params, &pt);
	std::cout << pt.get_child("path").get("dataFloder", "") << std::endl;
	boost::property_tree::write_json("default.json", pt);
	// Load Image Data to Memory.
	testLoad_Image(params);
	testpreProcess((const Params*) params);
	delete params;

	culaShutdown();
	return 0;
}

