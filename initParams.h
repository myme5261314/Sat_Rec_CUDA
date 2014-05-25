/*
 * initParams.h
 *
 *  Created on: 2014-5-4
 *      Author: peng
 */

#ifndef INITPARAMS_H_
#define INITPARAMS_H_

#include <iostream>
#include <string>
#include <boost/filesystem.hpp>
#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/json_parser.hpp>

#include "Params.h"

using namespace std;
using namespace boost::filesystem;
using namespace boost::property_tree;

/*
 * This is the function to Get data from the boost::PropertyTree variable.
 * It will lead to its default value when the queried entry doesn't exists.
 */
template<typename T>
void GetParameter(T &data, string query_str, ptree* pt);

void PropertyTree2Params(ptree *pt, Params *params);
void Params2PropertyTree(Params *params, ptree *pt);

Params* initParams(string path = "default.json");

#endif /* INITPARAMS_H_ */
