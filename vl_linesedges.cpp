#include "vl_linesedges.h"

#include <vector>

using namespace cv;

/*! \file vl_linesedges.cpp 
    \brief Line and edge extraction */

/*! * \addtogroup bimp
 * @{
 */

namespace bimp
{

/// \brief Finds lines and edges for one scale (wavelength) only. This is the basis of all multiscale functions.
///
/// \param lambda       Scale of the lines and edges
/// \param NO           Number of orientations
/// \param data         A KPData structure which will hold all the intermediate image data
///
/// N.B. this function assumes that the data parameter was processed by the \a gaborfilterbank function already
///
/// Returns a vector of objects
std::vector<LineEdge> linesedges( Type lambda, KPData &data, int NO )
{
    double t = (double)getTickCount();
    std::vector<LineEdge> edges;

    Type d = 0.6 * lambda;

    std::vector<Mat_<Type> > C_array = data.C_array;
    std::vector<Mat_<Type> > C_hat_array(NO);
    std::vector<Mat_<Type> > IL_array(NO), IC_array(NO);

    // Catch the error when the image has not been filtered yet
    if( C_array.size() != (unsigned) NO)
    {
        std::cerr << "Error! Wrong array size!" << std::endl;
        return std::vector<LineEdge>();
    }

    Type area = lambda*lambda * 25;

    // Iterate over all frequencies again and do inhibition
    #pragma omp parallel for schedule(dynamic,1) default(shared)
    for( int i=0; i<NO; i++)
    {
        int u = (i+NO/2)%NO;
        Type theta = u*CV_PI/NO;
        Type dsintheta = d*sin(theta);
        Type dcostheta = d*cos(theta);

        // calculate lateral and cross-orientation inhibition needed for lines and edges

        // Lateral inhibition (for lines and edges)
        Mat_<Type> C = C_array[i];
        Mat_<Type> IL(C.size(),0);

        int offset1 = round(dsintheta);
        int offset2 = round(dcostheta);
        Type *ilp, *firstp, *secondp;
        for(int row=abs(offset1)+1; row<IL.rows-abs(offset1)-1; row++)
        {
            ilp     = IL.ptr<Type>(row);
            firstp  = C.ptr<Type>(row+offset1);
            secondp = C.ptr<Type>(row-offset1);

            for(int col=abs(offset2)+1; col<IL.cols-abs(offset2)-1; col++)
            {
                Type val1, val2, result, larger;
                val1 = firstp[col+offset2];
                val2 = secondp[col-offset2];
                larger = val1 > val2 ? val1 : val2;
                result = val1 - val2;
                if(result <0) result = -result;
                result -= (val1+val2)/2; 
                if(result <0) result = 0;
                ilp[col] = result*4; // - (val1+val2)/2; 
            }
        }
        IL_array[i] = IL;

        // Cross-orientation inhibition (for lines and edges)
        Mat_<Type> IC(C.size(),0);
        C = C_array[i];
        Mat_<Type> C_other = C_array[(i+NO/2)%NO];

        offset1 = round(dsintheta*2);
        offset2 = round(dcostheta*2);
        Type *icp, *centrep;
        for(int row=abs(offset1)+1; row<IC.rows-abs(offset1)-1; row++)
        {
            icp = IC.ptr<Type>(row);
            centrep = C.ptr<Type>(row);
            firstp  = C_other.ptr<Type>(row+offset1);
            secondp = C_other.ptr<Type>(row-offset1);

            for(int col=abs(offset2)+1; col<IC.cols-abs(offset2)-1; col++)
            {
                Type val; 
                val =  firstp[col+offset2] + secondp[col-offset2] - 2*centrep[col];
                val =  val<0 ? 0 : val;
                icp[col] = val;
            }
        }
        IC_array[i] = IC;

    } // inhibition and interpolation loop

    // std::cout << "Inhibition: " << ((double)getTickCount()-t)/getTickFrequency() << std::endl;
    t = (double)getTickCount();

    #pragma omp parallel for schedule(dynamic,1) default(shared)
    for( int i=0; i<NO; i++)
        C_hat_array[i] = C_array[i]; // - IL_array[i]; // - IL_array[i];
    
    Mat_<Type> Ch(C_hat_array[0].size(), 0);
    for( int i=0; i<NO; i++)
        Ch += C_hat_array[i];

    // Type radius = lambda/4;
    Type radius = 1;
    
    Mat_<Type> LEloc(C_hat_array[0].size(), 0);
    Mat_<Type> LEori(C_hat_array[0].size(), 0);
    Mat_<unsigned char> LEtype(C_hat_array[0].size(), 0);
    
    t = (double)getTickCount();

    int ilambda = lambda;

    // Detect line and edge responses
    #pragma omp parallel for schedule(dynamic,1) default(shared)
    for(int row=ilambda; row<Ch.rows-ilambda; row++)
    {
        Type *centrep = Ch.ptr<Type>(row);
        // Type *firstp  = Ch.ptr<Type>(row-1);
        // Type *secondp = Ch.ptr<Type>(row+1);

        Type *lelocp = LEloc.ptr<Type>(row);
        Type *leorip = LEori.ptr<Type>(row);
        unsigned char *letypep = LEtype.ptr<unsigned char>(row);
        
        for(int col=ilambda; col<Ch.cols-ilambda; col++)
        {
            if( centrep[col] > area/2 )
            {
                int bestori = 0;
                Type bestvalue = 0;
                for(int i=0; i<NO; i++)
                    if(C_hat_array[i](row,col) > bestvalue)
                    {
                        bestvalue = C_hat_array[i](row,col);
                        bestori = i;
                    }
                if(bestvalue < 50) continue;

                Mat Chor = C_hat_array[bestori];
                
                // If not a maximum in the dominant orientation, forget about it!
                Type theta = bestori*CV_PI/NO;
                int offsetx = round(radius*cos(theta));
                int offsety = round(radius*sin(theta));

                // Look for maxima and minima in RO at the best orientation
                Mat_<Type> RO = data.RO_array[bestori];
                bool romax = ( RO(row,col)>RO(row+offsety,col+offsetx) && RO(row,col)>RO(row-offsety,col-offsetx) );
                bool romin = ( RO(row,col)<RO(row+offsety,col+offsetx) && RO(row,col)<RO(row-offsety,col-offsetx) );
                bool rozc  = ( RO(row+offsety, col+offsetx) * RO(row-offsety,col-offsetx) < 0 );

                // Look for maxima and minima in RE at the best orientation
                Mat_<Type> RE = data.RE_array[bestori];
                bool remax = ( RE(row,col)>RE(row+offsety,col+offsetx) && RE(row,col)>RE(row-offsety,col-offsetx) );
                bool remin = ( RE(row,col)<RE(row+offsety,col+offsetx) && RE(row,col)<RE(row-offsety,col-offsetx) );
                bool rezc  = ( RE(row+offsety, col+offsetx) * RE(row-offsety,col-offsetx) < 0 );
    
                // Second and third inhibition step, remove spurius stuff
                if( ( romax || romin || remax || remin ) != true ) continue;
                if( ( rezc  || rozc ) != true ) continue;
                 offsetx *=lambda/4;
                 offsety *=lambda/4;
                if( Chor.at<Type>(row,col) <= Chor.at<Type>(row+offsety,col+offsetx) || 
                    Chor.at<Type>(row,col) <= Chor.at<Type>(row-offsety,col-offsetx) ) continue;

                if( RE(row,col) < 1 && RO(row,col) < 1 && RE(row,col)>-1 && RO(row,col) > -1) continue; // noise

                Type xpos=0, ypos=0;
                if(0 && SUBPIXEL)
                {
                    Type offs = parabolaPeak(Chor.at<Type>(row-offsetx,col-offsety), 
                                             Chor.at<Type>(row,col), Chor.at<Type>(row+offsety,col+offsetx));
                    xpos += offs * cos(theta);
                    ypos += offs * sin(theta);
                }

                LineEdge le;
                le.response = Chor.at<Type>(row,col); //centrep[col];
                le.pt = Point2f(col+xpos,row+ypos);
                le.orientation = bestori*CV_PI/NO;
                le.i = bestori;
                le.size = lambda;
                // std::cout << col << " + " << xpos << ",  " << row << " + " << ypos<< std::endl;

                // Assign the line/edge type to the new element
                if( rozc && remax ) le.type = 1;    // positive line
                if( rozc && remin ) le.type = 2;    // negative line
                if( rezc && romax ) le.type = 3;    // positive edge
                if( rezc && romin ) le.type = 4;    // negative edge

                lelocp[col] = Chor.at<Type>(row,col); //centrep[col];
                leorip[col] = le.orientation;
                letypep[col] = le.type;
                // std::cout << "Ori: " << leorip[col] << ", Type: " << (int) letypep[col] << "-- "<<remax <<","<<remin<<","<<romax<<","<<romin<<":"<<rezc<<","<<rozc << std::endl;

                #pragma omp critical
                edges.push_back(le);
            }
        }
    }
    
    // Save intermediate map-like data to a data structure, in case we need it later
    data.C_array = C_array;
    data.C_hat_array = C_hat_array;
    data.Ch = Ch;
    data.IL_array = IL_array;
    data.IC_array = IC_array;
    data.LEloc = LEloc;
    data.LEori = LEori;
    data.LEtype = LEtype;
    data.lambda = lambda;

    // std::cout << Ch << std::endl;
    // std::cout << "Saving: " << ((double)getTickCount()-t)/getTickFrequency() << std::endl;
    // t = (double)getTickCount();

    return edges;
}

/// \brief Finds lines and edges for one scale only. Discard all the intermediate information 
///
/// \param img          Image to be filtered, with values between 0 and 1
/// \param lambda       Scale of lines and edges 
/// \param NO           Number of orientations
///
/// Returns a vector of \a LineEdge objects
std::vector<LineEdge> linesedges( Mat &img_orig, Type lambda, int NO )
{
    Mat img;
    
    // We need to convert the image to our internal type (usually double, defined in the header)
    if(img_orig.type() == DataType<Type>::type) 
        img = img_orig;
    else
        img_orig.convertTo( img, DataType<Type>::type );

    KPData data;

    gaborfilterbank(img,lambda,data,NO);
    return linesedges(lambda,data,NO);
}

/// \brief Finds multiscale lines and edges using image scaling. 
///
/// This function assumes that no filtering was done before, and that "datas" is empty. It should be used if you
/// are only interested in lines and edges, not keypoints. If you want to extract lines and edges after already
/// having found keypoints using keypoints_sc, you should use XXXX, which will reuse the filtering results and keep
/// the keypoints intact. 
///
/// Filtering is performed on a Gaussian pyramid, so this approach is fast. 
///
/// \param img          Image to be filtered, with values between 0 and 1
/// \param lambdas      A vector holding a list of scales to be processed
/// \param datas        A vector of KPData objects which will hold all the cell responses
/// \param NO           Number of orientations
///
/// Returns a vector of \a LineEdge objects
std::vector<LineEdge> linesedges_sc( Mat &img, std::vector<Type> lambdas, std::vector<KPData> &datas, int NO )
{
    double t = (double)getTickCount();

    std::vector<LineEdge> result;
    
    bool filtering;

    Type ratio = 1;
    Type pyrstep = 1;

    Mat myimg = img.clone();

    // The algorithm requires lambdas to come sorted in ascending order, so let's sort them
    std::sort( lambdas.begin(), lambdas.end());

    // Iterate over all the scales / wavelengths
    for(Type s = 1; s<=lambdas.size(); s++)
    {
        filtering = true;

        Type lambda    = lambdas[0];
        Type lambda2   = lambdas[s-1];
        ratio = lambda2 / lambda;

        // We usually perform line and edge extraction after extracting keypoints. If this is the case, then "datas"
        // will already contain complex cell responses, and we don't need to perform filtering again. So check
        // if a C_array exists and is filled with data
        // if(datas.C_array.size() == NO) 

        KPData data;

        // Time to resample the image, one step down in the pyramid
        while(lambda2/pyrstep > 6)
        {
            pyrDown(myimg,myimg);
            pyrstep *= 2;
        }

        std::vector<LineEdge> edges;

        gaborfilterbank( myimg, lambda2/pyrstep, data, NO);
        edges = linesedges( lambda2/pyrstep, data, NO);

        // scale the keypoints up again, for more accurate position
        for(unsigned i=0; i<edges.size(); i++)
        {
            edges[i].pt.x *= pyrstep;
            edges[i].pt.y *= pyrstep;
            edges[i].size *= pyrstep;
        }

        std::copy(edges.begin(), edges.end(), back_inserter(result));

        data.lambda = lambda2;
        data.pyrlevel = pyrstep;
        datas.push_back(data);

    } // loop over scales
    
    if(VERBOSE) std::cout << "Lines/edges: " << ((double)getTickCount()-t)/getTickFrequency() << " seconds. " << std::endl;

    return result;
}
/// \brief Finds multiscale lines and edges using image scaling. 
///
/// This function assumes that filtering was done already (e.g. by detecting keypoints first), and that "datas" 
/// contains all needed information. It should be used after having run keypoints_sc or similar. If you want to
/// extract lines and edges only, you should use linesedges_sc( Mat, std::vector<Type>, std::vector<KPData>, int NO ) .
///
/// Filtering is performed on a Gaussian pyramid, so this approach is fast. 
///
/// \param datas        A vector of KPData objects which will hold all the cell responses. This has to be filled already
///
/// Returns a vector of \a LineEdge objects
std::vector<LineEdge> linesedges_sc( std::vector<KPData> &datas )
{
    double t = (double)getTickCount();

    std::vector<LineEdge> result;
    
    for(Type s = 0; s<datas.size(); s++)
    {
        Type lambda    = datas[s].lambda;
        int pyrstep    = datas[s].pyrlevel;

        // Check whether complex cell responses are available
        if(datas[s].C_array.size() == 0)
        {
            std::cerr << "Error: complex cell responses not found at pyramid level " << pyrstep << std::endl;
            continue;
        }
        std::vector<LineEdge> edges;

        edges = linesedges( lambda, datas[s], datas[s].C_array.size());

        // scale the keypoints up again, for more accurate position
        for(unsigned i=0; i<edges.size(); i++)
        {
            edges[i].pt.x *= pyrstep;
            edges[i].pt.y *= pyrstep;
        }

        std::copy(edges.begin(), edges.end(), back_inserter(result));

    } // loop over scales
    
    if(VERBOSE) std::cout << "Lines/edges: " << ((double)getTickCount()-t)/getTickFrequency() << " seconds. " << std::endl;

    return result;
}

/// \brief Finds multiscale lines and edges using filters of increasing wavelength. 
///
/// This function assumes that no filtering was done before, and that "datas" is empty. It should be used if you
/// are only interested in lines and edges, not keypoints. If you want to extract lines and edges after already
/// having found keypoints using keypoints_fs, you should use XXXX, which will reuse the filtering results and keep
/// the keypoints intact. 
///
/// \param img          Image to be filtered, with values between 0 and 1
/// \param lambdas      A vector holding a list of scales to be processed
/// \param datas        A vector of KPData objects which will hold all the cell responses
/// \param NO           Number of orientations
///
/// Returns a vector of \a LineEdge objects
std::vector<LineEdge> linesedges_fs( Mat &img, std::vector<Type> lambdas, std::vector<KPData> &datas, int NO )
{
    double t = (double)getTickCount();

    std::vector<LineEdge> result;
    
    bool filtering;

    Mat myimg = img.clone();

    // The algorithm requires lambdas to come sorted in ascending order, so let's sort them
    std::sort( lambdas.begin(), lambdas.end());

    // Iterate over all the scales / wavelengths
    for(Type s = 1; s<=lambdas.size(); s++)
    {
        filtering = true;

        Type lambda    = lambdas[s-1];

        KPData data;

        std::vector<LineEdge> edges;

        gaborfilterbank( myimg, lambda, data, NO);
        edges = linesedges( lambda, data, NO);

        std::copy(edges.begin(), edges.end(), back_inserter(result));

        data.lambda = lambda;
        data.pyrlevel = s;
        datas.push_back(data);

    } // loop over scales
    
    if(VERBOSE) std::cout << "Lines/edges: " << ((double)getTickCount()-t)/getTickFrequency() << " seconds. " << std::endl;

    return result;
}
/// \brief Finds multiscale lines and edges using filters of increasing wavelength. 
///
/// This function assumes that filtering was done already (e.g. by detecting keypoints first), and that "datas" 
/// contains all needed information. It should be used after having run keypoints_sc or similar. If you want to
/// extract lines and edges only, you should use linesedges_fs( Mat, std::vector<Type>, std::vector<KPData>, int NO ) .
///
/// \param datas        A vector of KPData objects which will hold all the cell responses. This has to be filled already
///
/// Returns a vector of \a LineEdge objects
std::vector<LineEdge> linesedges_fs( std::vector<KPData> &datas )
{
    double t = (double)getTickCount();

    std::vector<LineEdge> result;
    
    for(Type s = 0; s<datas.size(); s++)
    {
        Type lambda    = datas[s].lambda;

        // Check whether complex cell responses are available
        if(datas[s].C_array.size() == 0)
        {
            std::cerr << "Error: complex cell responses not found at pyramid level " << s << std::endl;
            continue;
        }
        std::vector<LineEdge> edges;

        edges = linesedges( lambda, datas[s], datas[s].C_array.size());

        std::copy(edges.begin(), edges.end(), back_inserter(result));

    } // loop over scales
    
    if(VERBOSE) std::cout << "Lines/edges: " << ((double)getTickCount()-t)/getTickFrequency() << " seconds. " << std::endl;

    return result;
}

const float twosqrtpi=1.12838; 
const float onesqrtpi=0.56419; 
const float onesqrt2pi=0.39894;


} // namespace bimp


/*! * @} */
