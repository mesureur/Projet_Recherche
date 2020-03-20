#ifndef VL_COMMON_H
#define VL_COMMON_H

#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <fstream>
#include <string>
#include <vector>
#include <iostream>

/**
 * @namespace bimp
 * Documentation for bimp here. More docs for bimp here,
 * and down here.
 */

// setting this to true will enable subpixel localisation code
#ifndef SUBPIXEL 
#define SUBPIXEL true
#endif

#ifndef VERBOSE 
#define VERBOSE true
#endif

typedef double Type;
namespace bimp 
{

/*
typedef struct _Tabibi
{
    double time1;
    double time2;
    double time3;

    _Tabibi() { time1 = 0; time2 = 0; time3 = 0; };

} Tabibi;
*/

typedef struct _KPData
{
    Type lambda;
    int pyrlevel;
    int NO;

    std::vector<cv::Mat_<Type> > RO_array, RE_array;
    std::vector<cv::Mat_<Type> > C_array, S_array, D_array, C_hat_array;
    std::vector<cv::Mat_<Type> > IT_array, IR_array, IL_array, IC_array;
    cv::Mat_<Type> I, KPloc, LEloc, LEori, KD, KS, Ch;
    cv::Mat_<unsigned char> LEtype;

    _KPData() { lambda = -1; NO=0; pyrlevel=1; };

} KPData;


//Surcharge d'opérateur
//Dans la structure il faut 1 seul paramètre mais ça marche pas (error: "monFlux << data")
//A l'extérieur, il faut 2 paramètres mais :
//multiple definition of `bimp::operator<<(std::ostream&, bimp::_KPData const&)'

std::ostream &operator<<(std::ostream& os, KPData const& data);
std::istream &operator>>(std::istream& is, std::vector<int> tableau);

typedef struct _lineedge
{
    Type size;
    cv::Point2f pt;
    Type orientation;
    int i;
    Type response;

    unsigned char type;
    
    _lineedge() { size = -1; pt=cv::Point2f(-1,-1); orientation=0; type=0; response=0; }

} LineEdge;

void gaborfilterbank( const cv::Mat &img_orig, Type lambda, KPData &data, int NO );

Type parabolaPeak( Type val1, Type val2, Type val3, Type pos1=-1, Type pos2=0, Type pos3=1 );
std::vector<Type> makeLambdasLin( Type lambda_start, Type lambda_end, int numScales );
std::vector<Type> makeLambdasLog( Type lambda_start, Type lambda_end, int scalesPerOctave );

} // namespace bimp

#endif // VL_COMMON_H
