#include "opencv2/opencv.hpp"
#include "opencv2/imgproc/types_c.h"
#include "vl_keypoints.h"

using namespace std;
using namespace cv;
using namespace bimp;

int main(int argc, char** argv)
{
    VideoCapture cap;
    if(argc > 1)
    {
        cap.open(argv[1]); // open a video file
    }
    else
        cap.open(0); // open the default camera
        

    if(!cap.isOpened())  // check if we succeeded
    {
        std::cerr << "Can't open!" << std::endl;
        return -1;
    }

    vector<Type> lambdas = makeLambdasLin(20, 20, 1);

    double t = (double)getTickCount();

    Mat intensity;
    int framenum=0;
    Mat out;
    namedWindow("keypoints",1);

    for(;;)
    {
        Mat frame, frame2;
        cap >> frame; // get a new frame from camera
        if(frame.cols <=0 || frame.rows <= 0)
        {
            std::cout << "fillImages: Quitting after receiving an invalid frame." << std::endl;
            break;
        }
        framenum++;

        // resize for speed and convert to greyscale
        resize(frame, frame2, cv::Size(256,192), INTER_AREA );  
        cvtColor(frame2, intensity, CV_BGR2GRAY);
        
        // keypoint extraction and display
        std::vector<KPData> datas;
        std::vector<KeyPoint> points = keypoints(intensity,lambdas,datas,8,true);
        drawKeypoints(frame2, points, out, Scalar::all(255), static_cast<cv::DrawMatchesFlags>(4));

        imshow("keypoints", out);
        if(waitKey(1) >= 0) break;
    }
    
    double time = ((double)getTickCount()-t)/getTickFrequency();
    
    std::cout << "Execution time: " << time << std::endl;
    std::cout << framenum/time << " frames per second." << std::endl;

    return 0;
}
