#ifndef STOCKMAT_H
#define STOCKMAT_H


#include <iostream>
#include <fstream>

#include "vl_keypoints.h"
#include "vl_linesedges.h"

using namespace std;
using namespace cv;

void matwrite(const string& filename, const Mat& mat);
Mat matread(const string& filename);

#endif // STOCKMAT_H
