#include <iostream>
#include <fstream>
#include <string>
#include <vector>

#define NOMBRE_ITERATIONS 10

#include "vl_keypoints.h"
#include "vl_linesedges.h"

using namespace bimp;
using namespace std;
using namespace cv;

int main()
{
	/*

	On charge les différentes variables nécessaires pour faire fonctionner keypoints

	*/

	//Définition des différentes variables d'accueil
	Mat myimg;
	Type lambda;
	KPData data;
	int NO;

	// Load the saved matrix
	myimg = matread("raw.bin");

	//lambdas
	std::ifstream fichier_lambda("/home/jetson/Documents/keypoints-1.0/Variables_keypoints/lambda.txt");
	if(fichier_lambda)
		fichier_lambda >> lambda;
	else
		std::cout << "ERREUR: Impossible d'ouvrir le fichier lambdas en lecture." << std::endl;
	std::cout << "Voici lambda : " << lambda << std::endl;

	//NO
	ifstream fichier_no("/home/jetson/Documents/keypoints-1.0/Variables_keypoints/no.txt");
	if(fichier_no)
		fichier_no >> NO;
	else
	     	cout << "ERREUR: Impossible d'ouvrir le fichier NO en lecture." << endl;
	cout << "Voici NO : " << NO << std::endl;

	//data
	vector<Mat_<Type> > RO_array(NO), RE_array(NO);
    	vector<Mat_<Type> > C_array(NO);

		//RO_array
	RO_array[0] = matread("raw0.bin");
	RO_array[1] = matread("raw1.bin");
	RO_array[2] = matread("raw2.bin");
	RO_array[3] = matread("raw3.bin");
	RO_array[4] = matread("raw4.bin");
	RO_array[5] = matread("raw5.bin");
	RO_array[6] = matread("raw6.bin");
	RO_array[7] = matread("raw7.bin");
	data.RO_array = RO_array;

		//RE_array
	RE_array[0] = matread("raw8.bin");
	RE_array[1] = matread("raw9.bin");
	RE_array[2] = matread("raw10.bin");
	RE_array[3] = matread("raw11.bin");
	RE_array[4] = matread("raw12.bin");
	RE_array[5] = matread("raw13.bin");
	RE_array[6] = matread("raw14.bin");
	RE_array[7] = matread("raw15.bin");
	data.RE_array = RE_array;

		//C_array
	C_array[0] = matread("raw16.bin");
	C_array[1] = matread("raw17.bin");
	C_array[2] = matread("raw18.bin");
	C_array[3] = matread("raw19.bin");
	C_array[4] = matread("raw20.bin");
	C_array[5] = matread("raw21.bin");
	C_array[6] = matread("raw22.bin");
	C_array[7] = matread("raw23.bin");
	data.C_array = C_array;

	cout << "Le chargement c'est bien déroulé" << endl;



	/*

	Une fois que les différentes variables sont initialisées on fait tourner en boucle la fonction
std::vector<KeyPoint> keypoints( const Mat &img_orig, Type lambda, KPData &data, int NO=8);

	c'est la fonction que l'on veut faire tourner en boucle mais sans succès

	*/

	//Declaration des différentes variables
	vector<KeyPoint> points;
	double moyenne;
	double tab[NOMBRE_ITERATIONS];
	//double time1, time2, time3;

	//Fait tourner en boucle "keypoints"
	for (int i=0; i < NOMBRE_ITERATIONS; ++i){
		double tic = double(getTickCount());

		points = keypoints(myimg, lambda, data, NO);

		//Nouvelle fonction avec temps d'execution
		//points = keypoints(myimg, lambda, data, time1, time2, time3, NO);
        
		double toc = (double(getTickCount()) - tic) * 1000. / getTickFrequency();
		tab[i] = toc;
		
		//On reinitialise data pour repartir sur une base correcte
	        data = data;

		//cout << time1 << endl;
		//cout << time2 << endl;
		//cout << time3 << endl;
	}

	//On calcul la moyenne sur les différentes itérations
	for (int i=0; i < NOMBRE_ITERATIONS; ++i){
		moyenne += tab[i];
	}

	moyenne /= NOMBRE_ITERATIONS;

	cout << "Donc sur " << NOMBRE_ITERATIONS << " essais, le temps moyen d'execution de la fonction keypoints est de " << moyenne << " en millisecondes." << endl;



	/*

	Une fois que les différentes variables sont initialisées on fait tourner en boucle la fonction
std::vector<KeyPoint> keypoints( const Mat &img, std::vector<Type> lambdas, std::vector<KPData> &datas, int NO=8, bool scaling=true );

	Ce qui n'est pas la fonction qu'on veut faire tourner mais elle fonctionne

	*/

/*
	//Declaration des différentes variables
	vector<KeyPoint> points;
	vector<KPData> datas;
	double moyenne;
	double tab[NOMBRE_ITERATIONS];
	vector<Type> lambdas = makeLambdasLog(8, 64, 2);
	//double time1, time2, time3;

	//Initialisation de datas
	datas.push_back(data);


	//Fait tourner en boucle "keypoints"
	for (int i=0; i < NOMBRE_ITERATIONS; ++i){
		double tic = double(getTickCount());

		points = keypoints(myimg, lambdas, datas, NO);
        
		double toc = (double(getTickCount()) - tic) * 1000. / getTickFrequency();
		tab[i] = toc;
		
		//On reinitialise data pour repartir sur une base correcte
		datas.push_back(data);

		//cout << time1 << endl;
		//cout << time2 << endl;
		//cout << time3 << endl;
	}

	//On calcul la moyenne sur les différentes itérations
	for (int i=0; i < NOMBRE_ITERATIONS; ++i){
		moyenne += tab[i];
	}

	moyenne /= NOMBRE_ITERATIONS;

	cout << "Donc sur " << NOMBRE_ITERATIONS << " essais, le temps moyen d'execution de la fonction keypoints est de " << moyenne << " en millisecondes." << endl;
*/




	return 0;
}


