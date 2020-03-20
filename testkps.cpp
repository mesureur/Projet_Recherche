#include "vl_keypoints.h"
#include "vl_linesedges.h"

using namespace bimp;

int main( int argc, char **argv )
{
    if(argc < 2)
    {
        std::cerr << "Usage: testkps <filename>" << std::endl;
        return -1;
    }

    // Load the input image
    Mat img;
    Mat outimg; 
    img = imread(argv[1],0);

    // Select the scales we want to use (logarithmic spacing)
    std::vector<KeyPoint> points;
    std::vector<Type> lambdas = makeLambdasLog(8, 64, 2);

    // Extract keypoints at selected scales, 8 orientations
    std::vector<KPData> datas;
    points = keypoints(img,lambdas,datas,8,true);

    std::cout << std::endl << "Found " << points.size() << " points" << std::endl;
    
    // Draw the keypoints and save the image
    drawKeypoints(img, points, outimg, Scalar::all(255), static_cast<cv::DrawMatchesFlags>(4));
    imwrite("out.png", outimg);

    return 0;
}


