#ifndef LINESEDGES_H
#define LINESEDGES_H

#include "vl_common.h"

namespace bimp
{

std::vector<LineEdge> linesedges( Type lambda, KPData &data, int NO );
std::vector<LineEdge> linesedges( cv::Mat &img, Type lambda, int NO );
std::vector<LineEdge> linesedges_sc( cv::Mat &img, std::vector<Type> lambdas, std::vector<KPData> &datas, int NO );
std::vector<LineEdge> linesedges_sc( std::vector<KPData> &datas );
std::vector<LineEdge> linesedges_fs( cv::Mat &img, std::vector<Type> lambdas, std::vector<KPData> &datas, int NO );
std::vector<LineEdge> linesedges_fs( std::vector<KPData> &datas );

} // namespace bimp
#endif // LINESEDGES_H

