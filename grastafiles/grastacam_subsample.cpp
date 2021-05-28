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



//g++ -I /usr/local/include/opencv/ -L /usr/local/lib/ -lhighgui -lcvaux -lcxcore -L/opt/intel/composerxe-2011.4.191/mkl/lib/intel64  -Wl,--start-group -lmkl_intel_lp64 -lmkl_intel_thread -lmkl_core -Wl,--end-group -liomp5 -lpthread -lm grasta_cam_test_subsample.cpp -o gcts


///////////////////////////////////////////////////////////
///main
////////////////////////////////////////////////////////////

int main( int argc, char* argv[] ) {



//params assignment

srand(time(0));
float one=1.0f;
int oneinc=1;
float zero=0.0f;

//step size for subspace update. eta in original paper
float eta=.000001;

float rho=1; //some sort of step size for the w update (weights) and for the y update (lagrangian dual)

float *B; //guess for the subspace

float *tB; //unused?

float *pB; //unused?


float *v; //our guess for the image 

float *w; //weights of the subspace (which form the image's background)


float *Uw; //our guess for the background of the image
float *s; //our guess for the foreground of the image

//note: sgemv is just efficient matrix operation


int m,p,n,d;

//Image dimensions, height and width
int m=1.5*240;
int p=1.5*320;

//n in the paper. Vector dimension / pixels in image.
n=m*p;

//d in the paper. Subspace rank.
d=9;


//guess for subspace
B=(float*)malloc(n*d*sizeof(float));





//randomly assigned
for (ii=0;ii<n*d;ii++){
	B[ii]=rand();
}

w=(float*)malloc(d*sizeof(float));
v=(float*)malloc(n*sizeof(float));
Uw=(float*)malloc(n*sizeof(float));
s=(float*)malloc(n*sizeof(float));


float *tau;
tau=(float*)malloc(n*sizeof(float));

float  twork=0;
int lwork=-1;
int info;

//there is literally no other place where B could be made orthogonal, so this is probably it.



//a very complicated way of making B orthogonal.
// A
sgeqrf( &n, &d, B, &n, tau, &twork, &lwork, &info);	
lwork=(int) twork;	
//	printf("\n lwork=%d\n", lwork );		
float *work;
work=(float*)malloc(lwork*sizeof(float));
sgeqrf(&n, &d, B, &n, tau, work, &lwork, &info );

//multiply out the reflectors (B currently contains a mess which was returned by sgeqrf)
sorgqr(&n, &d, &d, B, &n, tau, work, &lwork, &info );

//at the end of this, we have factored B into QR, and set B = Q.
//maybe use better QR algorithm? room for improvement



//begin frontend

//cvNamedWindow( "selected location", 1 );
cvNamedWindow( "capture", 1 );
cvNamedWindow( "background?", 1 );
cvNamedWindow( "foreground?", 1 );


//create camera capture w/ error message
CvCapture* capture = cvCreateCameraCapture(1);
if(!capture){
	printf("failed to capture video from usb camera, trying built in camera\n");
	capture = cvCreateCameraCapture(CV_CAP_ANY);
	if(!capture){
		printf("failed to capture video\n");
		return(1);
	}
}   




CvFont font;
cvInitFont(&font, CV_FONT_HERSHEY_SIMPLEX, .5, .5, 0, 1, CV_AA);


IplImage* frame;
frame = cvQueryFrame(capture);

//a bunch of images which should show the decomposition of the image.
IplImage* outbw = cvCreateImage(cvGetSize(frame),IPL_DEPTH_8U,1);
IplImage* outg = cvCreateImage(cvGetSize(frame),IPL_DEPTH_32F,1);
IplImage* outgs = cvCreateImage(cvSize(p,m),IPL_DEPTH_32F,1);
IplImage* outgsb = cvCreateImage(cvSize(p,m),IPL_DEPTH_32F,1);
IplImage* outgsf = cvCreateImage(cvSize(p,m),IPL_DEPTH_32F,1);





//cvCopy(out,sframe,0);

//running total of learning rates (the model's adaptive based on user input)
//room for improvement: change learning rate more intelligently and remove user input
char etastring[40];

int c; //user input


//input/output params.
int names_count=0;

int classno=0;

double sample_percent=.1;
double  rm=double(RAND_MAX);

int use_number;
int* use_index;
use_index=(int*)malloc(m*sizeof(int));

int tcount=0;

//some sort of adaptive sample / subsample switch
int turbo=0;

float s_l1_norm=0;

while( 1 ) {
//	if (tcount++>4) break;	
	//get a new input frame (needs update)
	frame = cvQueryFrame(capture);
        if( !frame ) break;
	//make the frame grayscale and scale correctly
	cvCvtColor(frame,outbw,CV_BGR2GRAY);
	cvCvtScale(outbw,outg,.0039,0);//scale to 1/255
	cvResize(outg,outgs);

	v=(float*)outgs->imageData;

	sprintf(etastring,"eta = %.8f",eta);
	cvPutText(outgs,etastring , cvPoint(10, 40), &font, cvScalar(0, 0, 0, 0));
	cvShowImage("capture", outgs);

	//	
	rm=sample_percent*((double)RAND_MAX);
	use_number=0;	
	for (ii=0;ii<n;ii++){
		if (rand()<rm){ //this has a sample_percent chance of happening
			use_index[use_number]=ii;
			use_number++;			
		}
	}
//	fprintf(stderr,"use_number=%d\n",use_number);
	
	//perform step on B and then 
	if (turbo<5) {
		grasta_step (B,v,w,n,d,eta,rho,20);
	}
	else{
		//iterate via a random subsample of the image.
		grasta_step_subsample (B,v,w,n,d,eta,rho,40,use_index,use_number);
	}
	sgemv("N",&n,&d,&one,B,&n,w,&oneinc,&zero,Uw,&oneinc);
	s_l1_norm=0;	
	for (ii=0;ii<n;ii++){
		s[ii]=v[ii]-Uw[ii];
		if (fabs(s[ii])>.05){
			s_l1_norm ++;
			//s_l1_norm += fabs(s[ii]);
		}
	}
//	fprintf(stderr,"%f\n",s_l1_norm);
	if (s_l1_norm>n*.6){
		turbo=0;		
	}
	else{
		turbo++;
	}
/*	for(jj=0;jj<n;jj++){	
		g[jj]=g1[jj]-g2[jj];
	}*/
	outgsb->imageData = (char*) Uw;
	outgsb->imageDataOrigin = outgsb->imageData;
	cvShowImage( "background?", outgsb);

	outgsf->imageData = (char*)s;
	outgsf->imageDataOrigin = outgsf->imageData;
	cvNormalize(outgsf, outgsf,1,0,CV_MINMAX);
	cvShowImage( "foreground?", outgsf);

	//printf("%f\n",Uw[556]);
 	c = cvWaitKey(10);

	  //c = cvWaitKey(80);
        if( (char)c == 27 )
            break;
        switch( (char) c )
        {
        case 'm':
            sample_percent=sample_percent+.05;
		printf("sample percent up %.8f \n",sample_percent);
            break;
        case 'l':
            sample_percent=sample_percent-.05;
		printf("sample percent down %.8f \n",sample_percent);
            break;
        case 'u': //we have an option to increase the learning rate
            eta=3*eta/2;
		printf("eta up %.8f \n",eta);
            break;
        case 'd': //we have an option to decrease the learning rate.
                eta=2*eta/3;
		printf("eta down %.8f\n",eta );
;
            break;
        default:
            ;
        }
}




free(use_index);
}



