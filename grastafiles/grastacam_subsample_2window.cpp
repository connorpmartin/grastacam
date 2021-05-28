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
srand(time(0));
float one=1.0f;
int oneinc=1;
float zero=0.0f;

float dt=.000001;
float rho=1;
float *B,*tB,*pB,*x,*w,*bb,*ff;
int m,n,ii,jj;
int hh=1.5*240;
int ww=1.5*320;
m=hh*ww;
n=9;

B=(float*)malloc(m*n*sizeof(float));



for (ii=0;ii<m*n;ii++){
	B[ii]=rand();
}
w=(float*)malloc(n*sizeof(float));
x=(float*)malloc(m*sizeof(float));
bb=(float*)malloc(m*sizeof(float));
ff=(float*)malloc(m*sizeof(float));
float *tau;
tau=(float*)malloc(m*sizeof(float));

float  twork=0;
int lwork=-1;
int info;

sgeqrf( &m, &n, B, &m, tau, &twork, &lwork, &info);	
lwork=(int) twork;	
//	printf("\n lwork=%d\n", lwork );		
float *work;
work=(float*)malloc(lwork*sizeof(float));

sgeqrf(&m, &n, B, &m, tau, work, &lwork, &info );
sorgqr(&m, &n, &n, B, &m, tau, work, &lwork, &info );

//cvNamedWindow( "selected location", 1 );
cvNamedWindow( "capture", 1 );
cvNamedWindow( "background?", 1 );
//cvNamedWindow( "foreground?", 1 );


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

IplImage* outbw = cvCreateImage(cvGetSize(frame),IPL_DEPTH_8U,1);
IplImage* outg = cvCreateImage(cvGetSize(frame),IPL_DEPTH_32F,1);
IplImage* outgs = cvCreateImage(cvSize(ww,hh),IPL_DEPTH_32F,1);
IplImage* outgsb = cvCreateImage(cvSize(ww,hh),IPL_DEPTH_32F,1);
//IplImage* outgsf = cvCreateImage(cvSize(ww,hh),IPL_DEPTH_32F,1);

//cvCopy(out,sframe,0);

char dtstring[40];
int c;

int names_count=0;
int classno=0;

double sample_percent=.7;
double  rm=double(RAND_MAX);

int use_number;
int* use_index;
use_index=(int*)malloc(m*sizeof(int));

int tcount=0;

int turbo=0;
float ff_l1_norm=0;
while( 1 ) {
//	if (tcount++>4) break;	
	frame = cvQueryFrame(capture);
        if( !frame ) break;
	cvCvtColor(frame,outbw,CV_BGR2GRAY);
	cvCvtScale(outbw,outg,.0039,0);//scale to 1/255
	cvResize(outg,outgs);

	x=(float*)outgs->imageData;

	sprintf(dtstring,"dt = %.8f",dt);
	cvPutText(outgs,dtstring , cvPoint(10, 40), &font, cvScalar(0, 0, 0, 0));
	cvShowImage("capture", outgs);
	
	rm=sample_percent*((double)RAND_MAX);
	use_number=0;	
	for (ii=0;ii<m;ii++){
		if (rand()<rm){
			use_index[use_number]=ii;
			use_number++;			
		}
	}
//	fprintf(stderr,"use_number=%d\n",use_number);
	
	if (turbo<5) {
		grasta_step (B,x,w,m,n,dt,rho,20);
	}	
	else{
		grasta_step_subsample (B,x,w,m,n,dt,rho,40,use_index,use_number);
	}	
	sgemv("N",&m,&n,&one,B,&m,w,&oneinc,&zero,bb,&oneinc);
	ff_l1_norm=0;	
	for (ii=0;ii<m;ii++){
		ff[ii]=x[ii]-bb[ii];
		if (fabs(ff[ii])>.05){
			ff_l1_norm ++;
			//ff_l1_norm += fabs(ff[ii]);
		}
	}
//	fprintf(stderr,"%f\n",ff_l1_norm);
	if (ff_l1_norm>m*.6){
		turbo=0;		
	}
	else{
		turbo++;
	}
/*	for(jj=0;jj<m;jj++){	
		g[jj]=g1[jj]-g2[jj];
	}*/
	outgsb->imageData = (char*)bb;
	outgsb->imageDataOrigin = outgsb->imageData;
	cvShowImage( "background?", outgsb);

//	outgsf->imageData = (char*)ff;
//	outgsf->imageDataOrigin = outgsf->imageData;
//	cvNormalize(outgsf, outgsf,1,0,CV_MINMAX);
//	cvShowImage( "foreground?", outgsf);

	//printf("%f\n",bb[556]);
// 	c = cvWaitKey(10);

	  c = cvWaitKey(80);
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
        case 'u':
            dt=3*dt/2;
		printf("dt up %.8f \n",dt);
            break;
        case 'd':
                dt=2*dt/3;
		printf("dt down %.8f\n",dt );
;
            break;
        default:
            ;
        }
}




free(use_index);
}



