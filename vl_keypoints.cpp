#include <stdlib.h>
#include <stdio.h>
#include <cmath>
#include <omp.h>
#include <iostream>
#include <fstream>
#include <string>
#include <vector>

#include "vl_keypoints.h"

/*! \file vl_keypoints.cpp 
    \brief Keypoint extraction */

using namespace cv;
using namespace std;

namespace bimp
{


//Fonction "keypoints" adapté pour profilage (la vraie est juste en dessous)
std::vector<KeyPoint> keypoints( const Mat &img_orig, Type lambda, KPData &data, double& time1, double& time2, double& time3, int NO)
{

    //Affiche le temps initial + Prise de la référence de temps pour le reste du programme
    double stop0 = (double)getTickCount() / getTickFrequency();
    //std::cout << "Temps initial : " << stop0 << "\n";

    Mat img;

    // We need to convert the image to our internal type (usually double, defined in the header)
    if(img_orig.type() == DataType<Type>::type) 
        img = img_orig;
    else
        img_orig.convertTo( img, DataType<Type>::type );

    Type d = 0.6 * lambda;

    std::vector<Mat_<Type> > C_array = data.C_array;
    std::vector<Mat_<Type> > S_array(NO), D_array(NO); //, C_hat_array(NO);
    std::vector<Mat_<Type> > IT_array(NO), IR_array(NO); //, IL_array(NO), IC_array(NO);

    // Catch the error when the image has not been filtered yet
    if( C_array.size() != (unsigned) NO)
    {
        std::cerr << "Error! Wrong array size!" << std::endl;
        return std::vector<KeyPoint>();
    }

    // Iterate over all frequencies again and do inhibition
    #pragma omp parallel for schedule(dynamic,1) default(shared)
    for( int i=0; i<NO; i++)
    {
        Type theta = i*CV_PI/NO;
        Type dsintheta = d*sin(theta);
        Type dcostheta = d*cos(theta);

        Mat_<Type> C =       C_array[i];
        Mat_<Type> C_other = C_array[(i+NO/2)%NO];

        // Single-stopped complex cells
        Mat_<Type> S(C.size(),0);

        int offset1 = round(dcostheta);
        int offset2 = round(dsintheta);
        Type *sp, *firstp, *secondp;
        // for(int row=abs(offset1)+1; row<S.rows-abs(offset1)-1; row++)
        // {
        //     sp      = S.ptr<Type>(row);
        //     firstp  = C.ptr<Type>(row-offset1);
        //     secondp = C.ptr<Type>(row+offset1);

        //     for(int col=abs(offset2)+1; col<S.cols-abs(offset2)-1; col++)
        //     {
        //         Type val = firstp[col+offset2] - secondp[col-offset2];
        //         sp[col] = val>0 ? val : 0;
        //     }
        // }
        S_array[i] = S;

        // Double-stopped complex cells
        Mat_<Type> D(C.size(),0);

        offset1 = round(dcostheta*2);
        offset2 = round(dsintheta*2);
        Type *dp, *centrep;
        for(int row=abs(offset1)+1; row<D.rows-abs(offset1)-1; row++)
        {
            dp      = D.ptr<Type>(row);
            centrep = C.ptr<Type>(row);
            firstp  = C.ptr<Type>(row-offset1);
            secondp = C.ptr<Type>(row+offset1);
            
            for(int col=abs(offset2)+1; col<D.cols-abs(offset2)-1; col++)
            {
                Type val = centrep[col] - 0.5 * firstp[col+offset2] - 0.5 * secondp[col-offset2];
                dp[col]  = val>0 ? val : 0;
            }
        }

        D_array[i] = D;

        // Tangential inhibition
        Mat_<Type> IT(C.size(),0);

        offset1 = round(dsintheta);
        offset2 = round(dcostheta);
        Type *itp;
        for(int row=abs(offset1)+1; row<IT.rows-abs(offset1)-1; row++)
        {
            itp     = IT.ptr<Type>(row);
            centrep = C.ptr<Type>(row);
            firstp  = C.ptr<Type>(row+offset1);
            secondp = C.ptr<Type>(row-offset1);

            for(int col=abs(offset2)+1; col<IT.cols-abs(offset2)-1; col++)
            {
                Type val; 
                val =  firstp[col+offset2] - centrep[col];
                val =  val<0 ? 0 : val;
                val += secondp[col-offset2] - centrep[col];
                val =  val<0 ? 0 : val;
                itp[col] = val;
            }
        }
        IT_array[i] = IT;

        // Radial inhibition
        Mat_<Type> IR(C.size(),0);

        // ORIG: 2
        offset1 = round(dsintheta/2);
        offset2 = round(dcostheta/2);
        Type *irp;
        for(int row=abs(offset1)+1; row<IR.rows-abs(offset1)-1; row++)
        {
            irp = IR.ptr<Type>(row);
            centrep = C.ptr<Type>(row);
            firstp  = C_other.ptr<Type>(row+offset1);
            secondp = C_other.ptr<Type>(row-offset1);

            for(int col=abs(offset2)+1; col<IR.cols-abs(offset2)-1; col++)
            {
                Type val; 
                // ORIG: 4
                val =  centrep[col] - 16*firstp[col+offset2];
                val =  val<0 ? 0 : val;
                val += centrep[col] - 16*secondp[col-offset2];
                val =  val<0 ? 0 : val;
                // val = 0;
                irp[col] = val;
            }
        }
        IR_array[i] = IR;
        // imwrite("inr.png", IR/10);


    } // inhibition and interpolation loop



    //Affiche temps première boucle 
    double stop1 = (getTickCount()/getTickFrequency()) - stop0;
    time1 = stop1;
    //std::cout << "inhibition and interpolation loop : " << stop1 << "\n";



    // We still need to do some thresholding
    Type maxs = 0, maxd = 0;
    for( int i=0; i<NO; i++)
    {
        double min, max;
        minMaxLoc(S_array[i], &min, &max); 
        if(max > maxs) maxs = max;
        minMaxLoc(D_array[i], &min, &max);
        if(max > maxd) maxd = max;
    }

    // ORIGINAL: 0.05
    Type sthresh = 0.01* maxs;
    Type dthresh = 0.01* maxd;

    // sum up over orientations
    Mat_<Type> S_all (img.size(),0); 
    Mat_<Type> D_all (img.size(),0);
    Mat_<Type> IT_all(img.size(),0);
    Mat_<Type> IR_all(img.size(),0);
    Mat_<Type> maxori(img.size(),0);

    #pragma omp parallel for schedule(dynamic,1) default(shared) reduction(+: S_all, D_all, IT_all, IR_all)
    // il faudrait utiliser ceci dans la ligne du dessus : reduction(+: S_all, D_all, IT_all, IR_all) 
    for( int i=0; i<NO; i++)
    {
        // threshold small valued before summing up
        MatIterator_<Type> its = S_array[i].begin(), its_end = S_array[i].end();
        MatIterator_<Type> itd = D_array[i].begin(), itd_end = D_array[i].end();
        for(; its != its_end; ++its) *its = *its < sthresh ? 0 : *its;
        for(; itd != itd_end; ++itd) *itd = *itd < dthresh ? 0 : *itd;

        #pragma omp critical
        {
            S_all  += S_array[i]; 
            D_all  += D_array[i]; 
            IT_all += IT_array[i];
            IR_all += IR_array[i];
        }
    }


    //Applique un certain seuil et on somme les différentes inhibitions 
    double stop2 = (getTickCount() / getTickFrequency()) - stop1 - stop0;
    time2 = stop2;
    //std::cout << "Thresholding + Sum : " << stop2 << "\n";
    

    // add the two inhibition maps
    Mat_<Type> I = IT_all + IR_all;

    // perform inhibition
    Type g = 1;
    Mat_<float> KS = S_all - g*I;
    Mat_<float> KD = D_all - g*I;

    // extract keypoints
    std::vector<KeyPoint> points_s, points_d, points_all;

    Mat_<float> KDdil(KD.size()), KSdil(KS.size()), KPSloc, KPDloc;

    Matx<uchar,3,3> element(1,1,1,1,0,1,1,1,1);
    dilate(KD,KDdil,element);
    dilate(KS,KSdil,element);
    KPSloc = KS - KSdil;
    KPDloc = KD - KDdil;

    double mins, mind;
    double threshs, threshd;
    // ORIG: 0.04
    minMaxLoc(KS, &mins, &maxs); 
    threshs = 0.04 * maxs;
    minMaxLoc(KD, &mind, &maxd); 
    threshd = 0.04 * maxd;

    // find all the marked maxima and add the keypoints to the list
    // use an offset (=lambda) from image edges to avoid phantom keypoints caused by interference
    int off = lambda;
    for(int row=off; row<KPSloc.rows-off; row++)
        for(int col=off; col<KPSloc.cols-off; col++)
        {
            float xpos = col, ypos = row;
            if(KPSloc(row,col)>0 && KS(row,col)>threshs)    // found a keypoint
            {
                if(SUBPIXEL)
                {
                    xpos += parabolaPeak(KS(row,col-1), KS(row,col), KS(row,col+1));
                    ypos += parabolaPeak(KS(row-1,col), KS(row,col), KS(row+1,col));
                }

                KeyPoint kp( Point2f(xpos,ypos), lambda, maxori(row,col), KS(row,col),(int)lambda/16);
                points_s.push_back(kp);
            }
            else KPSloc(row,col) = 0;
            
            xpos = col; ypos = row;
            if(KPDloc(row,col)>0 && KD(row,col)>threshd)    // found a keypoint
            {
                if(SUBPIXEL)
                {
                    xpos += parabolaPeak(KD(row,col-1), KD(row,col), KD(row,col+1));
                    ypos += parabolaPeak(KD(row-1,col), KD(row,col), KD(row+1,col));
                }

                KeyPoint kp( Point2f(xpos,ypos), lambda, maxori(row,col), KD(row,col),(int)lambda/16);
                points_d.push_back(kp);
            }
            else KPDloc(row,col) = 0;
        }
    
    points_all = points_d;


    //Boucle_Keypoints_Found 
    double stop3 = (getTickCount() / getTickFrequency()) - stop2 - stop1 - stop0;
    time3 = stop3;
    //std::cout << "Find Keypoints : " << stop3 << "\n\n\n";


    // Save intermediate map-like data to a data structure, in case we need it later
    data.C_array = C_array;
    data.S_array = S_array;
    data.D_array = D_array;
    data.IT_array = IT_array;
    data.IR_array = IR_array;
    data.KPloc = KPDloc;
    data.I = I;
    data.KS = KS;
    data.KD = KD;
    data.lambda = lambda;
    data.NO = NO;

    return points_all;
}



/// \brief Finds keypoints for one scale (wavelength) only. This is the basis of all multiscale functions.
///
/// \param img          Image to be filtered, with values between 0 and 1
/// \param lambda       Scale of the keypoints
/// \param NO           Number of orientations
/// \param data         A KPData structure which will hold all the intermediate image data
///
/// N.B. this function assumes that the data parameter was processed by the vl_filters function already
///
/// Returns a vector of \a keypoint objects

std::vector<KeyPoint> keypoints( const Mat &img_orig, Type lambda, KPData &data, int NO)
{
    Mat img;

    // We need to convert the image to our internal type (usually double, defined in the header)
    if(img_orig.type() == DataType<Type>::type) 
        img = img_orig;
    else
        img_orig.convertTo( img, DataType<Type>::type );

    Type d = 0.6 * lambda;

    std::vector<Mat_<Type> > C_array = data.C_array;
    std::vector<Mat_<Type> > S_array(NO), D_array(NO); //, C_hat_array(NO);
    std::vector<Mat_<Type> > IT_array(NO), IR_array(NO); //, IL_array(NO), IC_array(NO);

    // Catch the error when the image has not been filtered yet
    if( C_array.size() != (unsigned) NO)
    {
        std::cerr << "Error! Wrong array size!" << std::endl;
        return std::vector<KeyPoint>();
    }

    // Iterate over all frequencies again and do inhibition
    #pragma omp parallel for schedule(dynamic,1) default(shared)
    for( int i=0; i<NO; i++)
    {
        Type theta = i*CV_PI/NO;
        Type dsintheta = d*sin(theta);
        Type dcostheta = d*cos(theta);

        Mat_<Type> C =       C_array[i];
        Mat_<Type> C_other = C_array[(i+NO/2)%NO];

        // Single-stopped complex cells
        Mat_<Type> S(C.size(),0);

        int offset1 = round(dcostheta);
        int offset2 = round(dsintheta);
        Type *sp, *firstp, *secondp;
        // for(int row=abs(offset1)+1; row<S.rows-abs(offset1)-1; row++)
        // {
        //     sp      = S.ptr<Type>(row);
        //     firstp  = C.ptr<Type>(row-offset1);
        //     secondp = C.ptr<Type>(row+offset1);

        //     for(int col=abs(offset2)+1; col<S.cols-abs(offset2)-1; col++)
        //     {
        //         Type val = firstp[col+offset2] - secondp[col-offset2];
        //         sp[col] = val>0 ? val : 0;
        //     }
        // }
        S_array[i] = S;

        // Double-stopped complex cells
        Mat_<Type> D(C.size(),0);

        offset1 = round(dcostheta*2);
        offset2 = round(dsintheta*2);
        Type *dp, *centrep;
        for(int row=abs(offset1)+1; row<D.rows-abs(offset1)-1; row++)
        {
            dp      = D.ptr<Type>(row);
            centrep = C.ptr<Type>(row);
            firstp  = C.ptr<Type>(row-offset1);
            secondp = C.ptr<Type>(row+offset1);
            
            for(int col=abs(offset2)+1; col<D.cols-abs(offset2)-1; col++)
            {
                Type val = centrep[col] - 0.5 * firstp[col+offset2] - 0.5 * secondp[col-offset2];
                dp[col]  = val>0 ? val : 0;
            }
        }

        D_array[i] = D;

        // Tangential inhibition
        Mat_<Type> IT(C.size(),0);

        offset1 = round(dsintheta);
        offset2 = round(dcostheta);
        Type *itp;
        for(int row=abs(offset1)+1; row<IT.rows-abs(offset1)-1; row++)
        {
            itp     = IT.ptr<Type>(row);
            centrep = C.ptr<Type>(row);
            firstp  = C.ptr<Type>(row+offset1);
            secondp = C.ptr<Type>(row-offset1);

            for(int col=abs(offset2)+1; col<IT.cols-abs(offset2)-1; col++)
            {
                Type val; 
                val =  firstp[col+offset2] - centrep[col];
                val =  val<0 ? 0 : val;
                val += secondp[col-offset2] - centrep[col];
                val =  val<0 ? 0 : val;
                itp[col] = val;
            }
        }
        IT_array[i] = IT;

        // Radial inhibition
        Mat_<Type> IR(C.size(),0);

        // ORIG: 2
        offset1 = round(dsintheta/2);
        offset2 = round(dcostheta/2);
        Type *irp;
        for(int row=abs(offset1)+1; row<IR.rows-abs(offset1)-1; row++)
        {
            irp = IR.ptr<Type>(row);
            centrep = C.ptr<Type>(row);
            firstp  = C_other.ptr<Type>(row+offset1);
            secondp = C_other.ptr<Type>(row-offset1);

            for(int col=abs(offset2)+1; col<IR.cols-abs(offset2)-1; col++)
            {
                Type val; 
                // ORIG: 4
                val =  centrep[col] - 16*firstp[col+offset2];
                val =  val<0 ? 0 : val;
                val += centrep[col] - 16*secondp[col-offset2];
                val =  val<0 ? 0 : val;
                // val = 0;
                irp[col] = val;
            }
        }
        IR_array[i] = IR;
        // imwrite("inr.png", IR/10);


    } // inhibition and interpolation loop

    cout << "Coucou n°1 " << endl;

    // We still need to do some thresholding
    Type maxs = 0, maxd = 0;
    for( int i=0; i<NO; i++)
    {
        double min, max;
        minMaxLoc(S_array[i], &min, &max); 
        if(max > maxs) maxs = max;
        minMaxLoc(D_array[i], &min, &max);
        if(max > maxd) maxd = max;
    }

    // ORIGINAL: 0.05
    Type sthresh = 0.01* maxs;
    Type dthresh = 0.01* maxd;

    // sum up over orientations
    Mat_<Type> S_all (img.size(),0); 
    Mat_<Type> D_all (img.size(),0);
    Mat_<Type> IT_all(img.size(),0);
    Mat_<Type> IR_all(img.size(),0);
    Mat_<Type> maxori(img.size(),0);

    #pragma omp parallel for schedule(dynamic,1) default(shared) reduction(+: S_all, D_all, IT_all, IR_all)
    // On a rajouter ça dans la ligne du dessus : reduction(+: S_all, D_all, IT_all, IR_all) 
    for( int i=0; i<NO; i++)
    {
        // threshold small valued before summing up
        MatIterator_<Type> its = S_array[i].begin(), its_end = S_array[i].end();
        MatIterator_<Type> itd = D_array[i].begin(), itd_end = D_array[i].end();
        for(; its != its_end; ++its) *its = *its < sthresh ? 0 : *its;
        for(; itd != itd_end; ++itd) *itd = *itd < dthresh ? 0 : *itd;

        #pragma omp critical
        {
	    cout << "Coucou n°1.5 " << endl;
            S_all  += S_array[i]; 
            D_all  += D_array[i]; 
            IT_all += IT_array[i];
            IR_all += IR_array[i];
	    cout << "Coucou n°2 " << endl;
        }
    }

    // add the two inhibition maps
    Mat_<Type> I = IT_all + IR_all;

    // perform inhibition
    Type g = 1;
    Mat_<float> KS = S_all - g*I;
    Mat_<float> KD = D_all - g*I;


    cout << "Coucou n°3 " << endl;

    // extract keypoints
    std::vector<KeyPoint> points_s, points_d, points_all;

    Mat_<float> KDdil(KD.size()), KSdil(KS.size()), KPSloc, KPDloc;

    Matx<uchar,3,3> element(1,1,1,1,0,1,1,1,1);
    dilate(KD,KDdil,element);
    dilate(KS,KSdil,element);
    KPSloc = KS - KSdil;
    KPDloc = KD - KDdil;

    double mins, mind;
    double threshs, threshd;
    // ORIG: 0.04
    minMaxLoc(KS, &mins, &maxs); 
    threshs = 0.04 * maxs;
    minMaxLoc(KD, &mind, &maxd); 
    threshd = 0.04 * maxd;

    // find all the marked maxima and add the keypoints to the list
    // use an offset (=lambda) from image edges to avoid phantom keypoints caused by interference
    int off = lambda;
    for(int row=off; row<KPSloc.rows-off; row++)
        for(int col=off; col<KPSloc.cols-off; col++)
        {
            float xpos = col, ypos = row;
            if(KPSloc(row,col)>0 && KS(row,col)>threshs)    // found a keypoint
            {
                if(SUBPIXEL)
                {
                    xpos += parabolaPeak(KS(row,col-1), KS(row,col), KS(row,col+1));
                    ypos += parabolaPeak(KS(row-1,col), KS(row,col), KS(row+1,col));
                }

                KeyPoint kp( Point2f(xpos,ypos), lambda, maxori(row,col), KS(row,col),(int)lambda/16);
                points_s.push_back(kp);
            }
            else KPSloc(row,col) = 0;
            
            xpos = col; ypos = row;
            if(KPDloc(row,col)>0 && KD(row,col)>threshd)    // found a keypoint
            {
                if(SUBPIXEL)
                {
                    xpos += parabolaPeak(KD(row,col-1), KD(row,col), KD(row,col+1));
                    ypos += parabolaPeak(KD(row-1,col), KD(row,col), KD(row+1,col));
                }

                KeyPoint kp( Point2f(xpos,ypos), lambda, maxori(row,col), KD(row,col),(int)lambda/16);
                points_d.push_back(kp);
            }
            else KPDloc(row,col) = 0;
        }
    
    points_all = points_d;

    // Save intermediate map-like data to a data structure, in case we need it later
    data.C_array = C_array;
    data.S_array = S_array;
    data.D_array = D_array;
    data.IT_array = IT_array;
    data.IR_array = IR_array;
    data.KPloc = KPDloc;
    data.I = I;
    data.KS = KS;
    data.KD = KD;
    data.lambda = lambda;
    data.NO = NO;

    return points_all;
}

/// \brief Finds keypoints for one scale only. Discard all the intermediate information 
///
/// \param img          Image to be filtered, with values between 0 and 1
/// \param lambda       Scale of the keypoints
/// \param NO           Number of orientations
///
/// Returns a vector of \a keypoint objects
std::vector<KeyPoint> keypoints( const Mat &img, Type lambda, int NO )
{
    KPData data;

    gaborfilterbank(img,lambda,data,NO);
    return keypoints(img,lambda,data,NO);
}

/// \brief Finds keypoints using image scaling. Filtering is performed on a Gaussian pyramid, so this approach is much faster. 
///
/// \param img          Image to be filtered, with values between 0 and 1
/// \param lambdas      A vector holding a list of scales to be processed
/// \param datas        A vector of KPData objects which will hold all the cell responses
/// \param NO           Number of orientations
///
/// Returns a vector of \a keypoint objects
std::vector<KeyPoint> keypoints_sc( const Mat &img, std::vector<Type> lambdas, std::vector<KPData> &datas, int NO )
{
    double t = (double)getTickCount();

    std::vector<KeyPoint> result;

    Type pyrstep = 1;

    Mat myimg = img.clone();

    // The algorithm requires lambdas to come sorted in ascending order, so let's sort them
    std::sort( lambdas.begin(), lambdas.end());

    // Iterate over all the scales / wavelengths
    for(Type s = 1; s<=lambdas.size(); s++)
    {
        Type lambda2   = lambdas[s-1];

        KPData data;

        // Time to resample the image, one step down in the pyramid
        while(lambda2/pyrstep > 6)
        {
            pyrDown(myimg,myimg);
            pyrstep *= 2;
        }

        std::vector<KeyPoint> points;

        gaborfilterbank( myimg, lambda2/pyrstep, data, NO);


	/*

	Stocke les attributs dans des fichiers texte appropriés
	Première vu les données étaient bien trop grandes pour être stockées dans des fichiers textes
	Donc finalement, on les stocke dans des fichiers binaires

	*/


  
	//On enregistre l'image
//Premiere version foireuse
/*
	std::string const image("/home/jetson/Documents/keypoints-1.0/Variables_keypoints/image.txt");
        std::ofstream fluxImage(image.c_str());
        if(fluxImage)
		fluxImage << myimg;
        else
        	std::cout << "ERREUR: Impossible d'ouvrir le fichier." << std::endl;
*/

//Deuxieme version plus performante (stockage binaire)
/*
	FileStorage fs("fs.yml", FileStorage::WRITE);
   	fs << "m" << myimg;
   	matwrite("raw.bin", myimg);

	//On enregistre les lambda
	std::string const lambda("/home/jetson/Documents/keypoints-1.0/Variables_keypoints/lambda.txt");
        std::ofstream fluxLambda(lambda.c_str());
        if(fluxLambda)
		fluxLambda << lambda2/pyrstep << std::endl;
        else
        	std::cout << "ERREUR: Impossible d'ouvrir le fichier." << std::endl;

	//On enregistre les data (KPData)
	std::string const multipledata("/home/jetson/Documents/keypoints-1.0/Variables_keypoints/data.txt");
        std::ofstream fluxData(multipledata.c_str());
        if(fluxData)
		fluxData << data << std::endl;
        else
        	std::cout << "ERREUR: Impossible d'ouvrir le fichier." << std::endl;

	//On enregistre le nombre NO
	std::string const no("/home/jetson/Documents/keypoints-1.0/Variables_keypoints/no.txt");
        std::ofstream fluxNo(no.c_str());
        if(fluxNo)
		fluxNo << NO << std::endl;
        else
        	std::cout << "ERREUR: Impossible d'ouvrir le fichier." << std::endl;
*/

        points = keypoints( myimg, lambda2/pyrstep, data, NO);

        // scale the keypoints up again, for more accurate position
        for(unsigned i=0; i<points.size(); i++)
        {
            points[i].pt.x *= pyrstep;
            points[i].pt.y *= pyrstep;
            points[i].size *= pyrstep;
        }

        std::copy(points.begin(), points.end(), back_inserter(result));

        data.lambda = lambda2;
        data.pyrlevel = pyrstep;
        datas.push_back(data);

    } // loop over scales
    
    if(VERBOSE) std::cout << "Keypoints: " << ((double)getTickCount()-t)/getTickFrequency() << " seconds. " << std::endl;

    return result;
}

/// \brief Finds keypoints using wavelets of increasing wavelength. All filtering is done on original size images
///
/// \param img          Image to be filtered, with values between 0 and 1
/// \param lambdas      A vector holding a list of scales to be processed
/// \param datas        A vector of KPData objects which will hold all the cell responses
/// \param NO           Number of orientations
///
/// Returns a vector of \a keypoint objects
std::vector<KeyPoint> keypoints_fs( const Mat &img, std::vector<Type> lambdas, std::vector<KPData> &datas, int NO )
{
    double t = (double)getTickCount();
    
    std::vector<KeyPoint> result;
    
    // Iterate over all the scales / wavelengths
    for(Type s = 1; s<=lambdas.size(); s++)
    {
        KPData data;

        std::vector<KeyPoint> points;
        Type lambda = lambdas[s-1]; 

        gaborfilterbank(img, lambda, data, NO);
        points = keypoints(img, lambda, data, NO);

        std::copy(points.begin(), points.end(), back_inserter(result));
    
        data.pyrlevel = 1;
        datas.push_back(data);
    } // loop over scales

    if(VERBOSE) std::cout << "Keypoints: " << ((double)getTickCount()-t)/getTickFrequency() << " seconds." << std::endl;

    return result;
}

/// \brief Finds multiscale keypoints using wavelets of increasing wavelength. 
///
/// This function acts as a front-end to keypoints_fs and keypoints_sc.
///
/// \param img          Image to be filtered, with values between 0 and 1
/// \param lambdas      A vector holding a list of scales to be processed
/// \param datas        A vector of KPData objects which will hold all the cell responses
/// \param NO           Number of orientations
/// \param scaling      Whether to scale images to Gaussian pyramid levels (default: true)
///
/// Returns a vector of \a keypoint objects
std::vector<KeyPoint> keypoints( const Mat &img, std::vector<Type> lambdas, std::vector<KPData> &datas, int NO, bool scaling)
{
    // bool stabilise = true;
    bool stabilise = false;

    std::sort(lambdas.begin(), lambdas.end());

    std::vector<KeyPoint> result;
    std::vector<KeyPoint> points;

    // Keypoint detection
    if(scaling)
        points = keypoints_sc( img, lambdas, datas, NO );
    else
        points = keypoints_fs( img, lambdas, datas, NO );

    // Stabilisation and scale selection
    // if(stabilise) 
    //     // result = stabilisekps(points, lambdas);
    //     result = stabilise4(points, lambdas);
    // else
        result = points;
   
    return result;
}

/// \brief Finds multiscale keypoints using wavelets of increasing wavelength. Discard all the intermediate information 
///
/// This function acts as a front-end to keypoints_fs and keypoints_sc.
///
/// \param img          Image to be filtered, with values between 0 and 1
/// \param lambdas      A vector holding a list of scales to be processed
/// \param NO           Number of orientations
/// \param scaling      Whether to scale images to Gaussian pyramid levels (default: true)
///
/// Returns a vector of \a keypoint objects
std::vector<KeyPoint> keypoints( const Mat &img, std::vector<Type> lambdas, int NO, bool scaling){
  std::vector<KPData> datas;
  return keypoints(img,lambdas,datas,NO,scaling);
}

bool gt_kp_fun( KeyPoint j, KeyPoint k ) { return (j.response > k.response); }


/// \brief Finds multiscale keypoints using wavelets of increasing wavelength. 
///
/// This function acts as a front-end to keypoints_fs and keypoints_sc. It discards all the intermediate data (cell responses) and
/// only returns the keypoint positions.
///
/// \param img          Image to be filtered, with values between 0 and 1
/// \param lambdas      A vector holding a list of scales to be processed
/// \param NO           Number of orientations
/// \param scaling      Whether to scale images to Gaussian pyramid levels (default: true)
///
/// Returns a vector of \a keypoint objects
std::vector<KeyPoint> keypoints( Mat &img, std::vector<Type> lambdas, int NO, bool scaling )
{
    std::vector<KeyPoint> result;
    std::vector<KPData> datas;

    return keypoints( img, lambdas, datas, NO, scaling );
}

} // namespace bimp
