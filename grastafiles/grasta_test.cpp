#include <opencv/highgui.h>
#include <string.h>
#include <opencv/cv.h>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <ctime>
#include <time.h>
#include "math.h"
#include "mkl.h"
#include "mkl_blas.h"
#include <stdlib.h>
#include <stdio.h>
#include "larb.h"
#include "grasta.h"

int main( int argc, char* argv[] ) {

srand(time(0));
float one=1.0f;
int oneinc=1;
float zero=0.0f;

float eta=.000001; //step size for subspace update. eta in original paper

float rho=1; //some sort of step size for the w update (weights) and for the y update (lagrangian dual). rho in original paper

float *U; //guess for the subspace. n x d

float *tB; //unused?

float *pB; //unused?

float *v; //our guess for the image. n x 1

float *w; //weights of the subspace (which form the image's background). d x 1

float *Uw; //our guess for the background of the image. n x 1
float *s; //our guess for the foreground of the image. n x 1

//note: sgemv is just efficient matrix operation and sgeqrf is just QR factorization using householder

//Image dimensions, height and width
int m,p,n,d;

m=1.5*240;
p=1.5*320;


//Vector dimension (pixels) in image.
n=m*p;

//Subspace rank.
d=9;

int numIters = 60; //total number of frames fed into the model.

float *tau; //out scale factors of householder reflections. n x 1
tau=(float*)malloc(n*sizeof(float));

float  twork=0; //out for sgeqrf
int lwork=-1; //out dimension of twork
int info; // out for sgeqrf

//generate underlying nm by d subspace for images.
float *data = (float*)malloc(n*numIters*sizeof(float));

//guess for subspace
U=(float*)malloc(n*d*sizeof(float));

for(int i=0;i<n*d;++i){
    data[i] = rand();
}


//randomly assigned initial subspace
for (ii=0;ii<n*d;ii++){
	U[ii]=rand();
}

//a very complicated way of making U orthogonal.
sgeqrf( &n, &d, data, &n, tau, &twork, &lwork, &info);	
lwork=(int) twork;	
//	printf("\n lwork=%d\n", lwork );		
float *work;
work=(float*)malloc(lwork*sizeof(float));
sgeqrf(&n, &d, data, &n, tau, work, &lwork, &info );

//multiply out the reflectors (data currently contains a mess which was returned by sgeqrf)
sorgqr(&n, &d, &d, data, &n, tau, work, &lwork, &info );

//a very complicated way of making U orthogonal.
sgeqrf( &n, &d, U, &n, tau, &twork, &lwork, &info);	
lwork=(int) twork;	
//	printf("\n lwork=%d\n", lwork );		
float *work;
work=(float*)malloc(lwork*sizeof(float));
sgeqrf(&n, &d, U, &n, tau, work, &lwork, &info );

//multiply out the reflectors (U currently contains a mess which was returned by sgeqrf)
sorgqr(&n, &d, &d, U, &n, tau, work, &lwork, &info );


//data is now a n by d orthogonal matrix
//to generate our v we take random subsamples of this subspace
//represented by multiplying by d by 1 random weight matrix.


//U is our initial random guess.

float *rand_weights = (float*)malloc(d*sizeof(float));

for(int i=0;i<numIters;i++){

    for(int j=0;j<d;j++){
        rand_weights[j] = rand();
    }
    //TODO: add foreground and noise
    //generate v as a random element of the subspace from data.
    sgemv("N",&n,&d,&one,data,&n,rand_weights,&oneinc,&zero,v,&oneinc);

    grasta_step (U,v,w,n,d,eta,rho,20);

    //output some stuff

    //probably output current guess

}

//output final guess for video


free(U);
free(data);
free(tau);
free(v);
free(w);
free(s);
free(Uw);
}