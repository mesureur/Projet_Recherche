#include <string>

#include "stock_mat.h"
#include "vl_common.h"

/*! \file vl_common.cpp 
    \brief Common functions for keypoints and lines and edges */

using namespace cv;
using namespace std;

namespace bimp
{

//surcharge d'opérateur<< et >>
ostream &operator<<(std::ostream& os, KPData const& data)
{

//Premiere version naze
/*
	//os << data.RO_array.size();
	for (auto i : data.RO_array)
	{
	    os << i;
	}
	os << std::endl;

	//os << data.RE_array.size();
	for (auto i : data.RE_array)
	{
	    os << i;
	}
	os << std::endl;

	//os << data.C_array.size();
	for (auto i : data.C_array)
	{
	    os << i;
	}
	os << std::endl;
*/

//Deuxieme version plus performante
	//manuellement on stocke les 8 différentes orientations
//RO_array
	FileStorage fs0("fs0.yml", FileStorage::WRITE);
   	fs0 << "m" << data.RO_array;
   	matwrite("raw0.bin", data.RO_array[0]);

	FileStorage fs1("fs1.yml", FileStorage::WRITE);
   	fs1 << "m" << data.RO_array;
   	matwrite("raw1.bin", data.RO_array[1]);

	FileStorage fs2("fs2.yml", FileStorage::WRITE);
   	fs2 << "m" << data.RO_array;
   	matwrite("raw2.bin", data.RO_array[2]);

	FileStorage fs3("fs3.yml", FileStorage::WRITE);
   	fs3 << "m" << data.RO_array;
   	matwrite("raw3.bin", data.RO_array[3]);

	FileStorage fs4("fs4.yml", FileStorage::WRITE);
   	fs4 << "m" << data.RO_array;
   	matwrite("raw4.bin", data.RO_array[4]);

	FileStorage fs5("fs5.yml", FileStorage::WRITE);
   	fs5 << "m" << data.RO_array;
   	matwrite("raw5.bin", data.RO_array[5]);

	FileStorage fs6("fs6.yml", FileStorage::WRITE);
   	fs6 << "m" << data.RO_array;
   	matwrite("raw6.bin", data.RO_array[6]);

	FileStorage fs7("fs7.yml", FileStorage::WRITE);
   	fs7 << "m" << data.RO_array;
   	matwrite("raw7.bin", data.RO_array[7]);


//RE_array
	FileStorage fs8("fs8.yml", FileStorage::WRITE);
   	fs8 << "m" << data.RE_array;
   	matwrite("raw8.bin", data.RE_array[0]);

	FileStorage fs9("fs9.yml", FileStorage::WRITE);
   	fs9 << "m" << data.RE_array;
   	matwrite("raw9.bin", data.RE_array[1]);

	FileStorage fs10("fs10.yml", FileStorage::WRITE);
   	fs10 << "m" << data.RE_array;
   	matwrite("raw10.bin", data.RE_array[2]);

	FileStorage fs11("fs11.yml", FileStorage::WRITE);
   	fs11 << "m" << data.RE_array;
   	matwrite("raw11.bin", data.RE_array[3]);

	FileStorage fs12("fs12.yml", FileStorage::WRITE);
   	fs12 << "m" << data.RE_array;
   	matwrite("raw12.bin", data.RE_array[4]);

	FileStorage fs13("fs13.yml", FileStorage::WRITE);
   	fs13 << "m" << data.RE_array;
   	matwrite("raw13.bin", data.RE_array[5]);

	FileStorage fs14("fs14.yml", FileStorage::WRITE);
   	fs14 << "m" << data.RE_array;
   	matwrite("raw14.bin", data.RE_array[6]);

	FileStorage fs15("fs15.yml", FileStorage::WRITE);
   	fs15 << "m" << data.RE_array;
   	matwrite("raw15.bin", data.RE_array[7]);

	
//C_array
	FileStorage fs16("fs16.yml", FileStorage::WRITE);
   	fs16 << "m" << data.C_array;
   	matwrite("raw16.bin", data.C_array[0]);

	FileStorage fs17("fs17.yml", FileStorage::WRITE);
   	fs17 << "m" << data.C_array;
   	matwrite("raw17.bin", data.C_array[1]);

	FileStorage fs18("fs10.yml", FileStorage::WRITE);
   	fs18 << "m" << data.C_array;
   	matwrite("raw18.bin", data.C_array[2]);

	FileStorage fs19("fs19.yml", FileStorage::WRITE);
   	fs19 << "m" << data.C_array;
   	matwrite("raw19.bin", data.C_array[3]);

	FileStorage fs20("fs20.yml", FileStorage::WRITE);
   	fs20 << "m" << data.C_array;
   	matwrite("raw20.bin", data.C_array[4]);

	FileStorage fs21("fs21.yml", FileStorage::WRITE);
   	fs21 << "m" << data.C_array;
   	matwrite("raw21.bin", data.C_array[5]);

	FileStorage fs22("fs22.yml", FileStorage::WRITE);
   	fs22 << "m" << data.C_array;
   	matwrite("raw22.bin", data.C_array[6]);

	FileStorage fs23("fs23.yml", FileStorage::WRITE);
   	fs23 << "m" << data.C_array;
   	matwrite("raw23.bin", data.C_array[7]);

	return os;
}


std::istream &operator>>(std::istream& is, std::vector<int> tableau) 
{
	//On stocke l'ensemble des valeurs dans un tableau
	char a;
	int compteur = 0;
	//On récupère la taille de la matrice
	is.seekg(0,std::ios::end);
	int size = is.tellg();
	//On se replace au début du fichier
	is.seekg(0,std::ios::beg);

	for (int i=0; i<size; ++i)
	{
		is.get(a);
		//std::cout << a << std::endl;
		
		if(a=='[');
		else if (a==']');
		else if (a==' ');
		else if (a==',');
		else
		{
			std::cout << a << std::endl;
			tableau.push_back(a);
			std::cout << tableau[compteur] << std::endl;
			++compteur;
		}	
	}		

	return is;
}

/// \brief Runs a bank of Gabor filters. This function is called by all keypoint and line/edge algorithms
///
/// \param img          Image to be filtered, with values between 0 and 1
/// \param lambda       Scale of the keypoints
/// \param NO           Number of orientations
/// \param data         A KPData structure which will hold all the intermediate image data
void gaborfilterbank( const Mat &img_orig, Type lambda, KPData &data, int NO )
{
    // double t = (double)getTickCount();
    Mat img;

    // We need to convert the image to our internal type (usually double, defined in the header)
    if(img_orig.type() == DataType<Type>::type) 
        img = img_orig;
    else
        img_orig.convertTo( img, DataType<Type>::type );

    // std::cout << "Conversion: " << ((double)getTickCount()-t)/getTickFrequency() << std::endl;
    // t = (double)getTickCount();
    
    Type gamma = 0.5;
    // Type gamma = 1;
    Type phi = -CV_PI/2;
    Type sigma = lambda * 0.56;

    std::vector<Mat_<Type> > RO_array(NO), RE_array(NO);
    std::vector<Mat_<Type> > C_array(NO);

    // set the filter dimensions to something reasonable (m x lambda)
    unsigned filtersize = lambda*5;
    // ensure odd size of the filter
    if(filtersize%2 == 0) filtersize += 1; 
    Type offsetf = floor(filtersize/2);
    
    // iterate over orientations
    #pragma omp parallel for schedule(dynamic,1) default(shared)
    for( int i=0; i<NO; i++)
    {
        Type theta = i*CV_PI/NO;
        Type sintheta = sin(theta);
        Type costheta = cos(theta);
        Type sigmasq = sigma * sigma;
        Type evensum = 0;

        Mat_<Type> kernel_odd(filtersize, filtersize);
        Mat_<Type> kernel_evn(filtersize, filtersize);
        Mat_<Type> RO(img.size());
        Mat_<Type> RE(img.size());
        Mat_<Type> C(img.size());

        // create the wavelet we will be using
        for(int row=0; row<kernel_odd.rows && row<kernel_evn.rows; row++)
        {
            for(int col=0; col<kernel_odd.cols && col<kernel_evn.cols; col++)
            {
                int x = col - offsetf;
                int y = row - offsetf;

                Type env = exp( -1/(2*sigmasq) * ( (x*costheta + y*sintheta) * (x*costheta + y*sintheta)
                            + gamma * (y*costheta - x*sintheta) * (y*costheta - x*sintheta) ) );
                Type freqfactor = 2 * CV_PI * (x*costheta + y*sintheta)/lambda;

                kernel_evn(row,col) = env * cos(freqfactor);
                kernel_odd(row,col) = env * cos(freqfactor+phi);
                evensum += kernel_evn(row,col);
            }
        }
        // Remove the DC component of the even kernel
        kernel_evn -= evensum/(filtersize*filtersize);
        
        // std::cout << evensum << std::endl;

        // Filter the image  -- flip the kernels because filter2D does correlation, not convolution
        flip(kernel_odd, kernel_odd, -1);
        flip(kernel_evn, kernel_evn, -1);

        filter2D( img, RO, RO.depth(), kernel_odd);
        filter2D( img, RE, RE.depth(), kernel_evn);

        // calculate the complex cell response
        magnitude(RE,RO,C);

        RO_array[i] = RO;
        RE_array[i] = RE;
        C_array[i] = C;
        // imwrite("ro.png", RO);
        // imwrite("re.png", RE);
        // imwrite("c.png", C);

    } // filtering loop

    // std::cout << "Filtering: " << ((double)getTickCount()-t)/getTickFrequency() << std::endl;
    // t = (double)getTickCount();

    data.RO_array = RO_array;
    data.RE_array = RE_array;
    data.C_array = C_array;
}

/// \brief Fits a parabola to three points and estimates the peak position
///
/// This helper function is used to localise the peaks of hypercomplex cell responses with subpixel precision. 
/// Since the parabola we're fitting is linearly separable, we can do this separately in the x and y dimensions
Type parabolaPeak( Type val1, Type val2, Type val3, Type pos1, Type pos2, Type pos3 )
{
    Type result;

    Mat_<Type> xs(3,3), ys(3,1), factors(3,1);
    ys(0,0) = val1;
    ys(1,0) = val2;
    ys(2,0) = val3;

    xs(0,0) = pos1 * pos1;
    xs(0,1) = pos1;
    xs(0,2) = 1;
    xs(1,0) = pos2 * pos2;
    xs(1,1) = pos2;
    xs(1,2) = 1;
    xs(2,0) = pos3 * pos3;
    xs(2,1) = pos3;
    xs(2,2) = 1;

    solve( xs, ys, factors );
    
    result = -factors(1,0) / (factors(0,0)*2);

    return result;
}

/// \brief Creates a vector of linearly spaced wavelengths (lambda) for multiscale analysis  
///
/// \param lambda_start Initial wavelength of the gabor filters
/// \param lambda_end   Ending wavelength of the gabor filters
/// \param numScales    Number of scales (linearly spread between \a lambda_start and \a lambda_end)
///
/// Returns a vector of wavelengths
std::vector<Type> makeLambdasLin( Type lambda_start, Type lambda_end, int numScales )
{
    std::vector<Type> lambdas;
    Type lambda_step = 1;
    int NS = numScales;
    
    if(NS>1) lambda_step = (lambda_end-lambda_start)/(NS-1);
    
    for(Type s = 1; s<=NS; s++)
    {
        Type lambda = lambda_start + (s-1) * lambda_step;
        lambdas.push_back(lambda);
    }

    return lambdas;
}

/// \brief Creates a vector of logarithmically spaced wavelengths (lambda) for multiscale analysis  
///
/// \param lambda_start     Initial wavelength of the gabor filters
/// \param lambda_end       Ending wavelength of the gabor filters
/// \param scalesPerOctave  Number of scales per octave
///
/// Returns a vector of wavelengths
std::vector<Type> makeLambdasLog( Type lambda_start, Type lambda_end, int scalesPerOctave )
{
    std::vector<Type> lambdas;
    
    Type multiplier = pow(2.,1./scalesPerOctave);

    for(Type lambda=lambda_start; lambda<=lambda_end*1.05; lambda*=multiplier)
    {
        lambdas.push_back(lambda);
    }

    return lambdas;
}

} // namespace bimp
