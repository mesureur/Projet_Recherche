#ifndef KEYPOINTS_H
#define KEYPOINTS_H

#include "vl_common.h"
#include "stock_mat.h"

using namespace cv;

namespace bimp
{

//Ma fonction à moi ;)
std::vector<KeyPoint> keypoints( const Mat &img_orig, Type lambda, KPData &data, double& time1, double& time2, double& time3, int NO);

std::vector<KeyPoint> keypoints( const Mat &img_orig, Type lambda, KPData &data, int NO=8);
std::vector<KeyPoint> keypoints( const Mat &img, Type lambda, int NO=8 );

std::vector<KeyPoint> keypoints( const Mat &img, std::vector<Type> lambdas, int NO=8, bool scaling=true);
std::vector<KeyPoint> keypoints( const Mat &img, std::vector<Type> lambdas, std::vector<KPData> &datas, int NO=8, bool scaling=true );

std::vector<KeyPoint> keypoints_fs( const Mat &img, std::vector<Type> lambdas, std::vector<KPData> &datas, int NO=8 );

std::vector<KeyPoint> keypoints_sc( const Mat &img_orig, std::vector<Type> lambdas, std::vector<KPData> &datas, int NO=8);

}   // namespace bimp

#endif // KEYPOINTS_H
