//#include "cycle.h"



void grasta_step (float* B, float* v, float* w, int n, int d, float eta,float rho, int maxiter){

	float one=1.0f;
	int oneinc=1;
	float zero=0.0f;
	int ii,jj;
	//float* w=(float*)malloc(n*sizeof(float));
	float* s=(float*)calloc(n,sizeof(float));
	float* y=(float*)calloc(n,sizeof(float));
	float* g1=(float*)malloc(n*sizeof(float));
	float* uw=(float*)malloc(n*sizeof(float));
	float* Ug2=(float*)malloc(n*sizeof(float));
	float* g2=(float*)malloc(n*sizeof(float));
	float* g=(float*)malloc(n*sizeof(float));
	float sigma;
	float normg;
	float normw;
	float cs;
	float ss;


	//update w,s,y
	larb_orthogonal_alt(B,n,d,v,w,s,y,rho,maxiter);


	//3.8: calculate Γ_1 (stored in g1)

	//uw = Uw
	sgemv("N",&n,&d,&one,B,&n,w,&oneinc,&zero,uw,&oneinc); 
	for(jj=0;jj<m;jj++){
		g1[jj]=y[jj]+rho*(uw[jj]+s[jj]-v[jj]);
	}

	//3.9: calculate Γ_2 (stored in g2)

	sgemv("T",&n,&d,&one,B,&n,g1,&oneinc,&zero,g2,&oneinc);


	//3.10: calculate Γ. 

	//g2 = U(Γ_2)
	sgemv("N",&n,&d,&one,B,&n,g2,&oneinc,&zero,Ug2,&oneinc);

	//final calculation of Γ (stored in g)
	for(jj=0;jj<n;jj++){
		g[jj]=g1[jj]-Ug2[jj];
	}

	//calcuate sigma (between 3.11 and 3.12)
	normg=sdot(&n,g,&oneinc,g,&oneinc);
	normg=sqrt(normg);

	normw=sdot(&d,w,&oneinc,w,&oneinc);
	normw=sqrt(normw);

	sigma=normg*normw;


	//note that this equation distributes the 1/|w| 
	//so we get a |w|^2 term in cs, and a 1/(|w||Γ|) = 1/sigma
	//term in ss.

	cs=(cos(eta*sigma)-1)/(normw*normw);
	ss=sin(eta*sigma)/sigma;

	//reuse g1 as a temporary matrix.
	//This is the term inside of the large parenthesis in 3.13.
	//but with the 1/|w| distributed in.
	for(jj=0;jj<n;jj++){
		g1[jj]=cs*uw[jj]-ss*g[jj];
	}

	//calculate U + g1 * w^T
	//note that g1 is the temporary matrix from the last line. 
	for(ii=0;ii<n;ii++){//row 
		for(jj=0;jj<d;jj++){//column
			B[jj*n+ii]=B[jj*n+ii]+g1[ii]*w[jj];
		}
	}

	free (s);
	free(y);
	free(g1);
	free(uw);
	free(Ug2);
	free(g2);
	free(g);
}























void grasta_step_subsample(float* B, float* v, float* w, int n, int d, float eta,float rho, int maxiter, int* use_index, int use_number){

	float one=1.0f;
	int oneinc=1;
	float zero=0.0f;
	int ii,jj;
	float* tB;
	float* smallv;
	float* smallB;
	float* pismallB;
//	float* w=(float*)malloc(n*sizeof(float));
	float* s=(float*)calloc(use_number,sizeof(float));
	float* y=(float*)calloc(use_number,sizeof(float));


	float* g1=(float*)malloc(use_number*sizeof(float));
	float* uw=(float*)malloc(use_number*sizeof(float));
	float* Uw=(float*)malloc(n*sizeof(float));
	float* Ug2=(float*)malloc(d*sizeof(float));
	float* g2=(float*)malloc(n*sizeof(float));
	float* g=(float*)malloc(n*sizeof(float));
	float sigma;
	float normg;
	float normw;
	float cs;
	float ss;
	float scale=0;
	float fuse_number=(float)use_number;


//	ticks t0 = getticks();

//    fprintf(stderr,"0 in grasta step \n");

	//This is like grasta_step, but we have a mask array use_index (int array with indices)


	//allocate memory for our subsample
	smallv=(float*)malloc(use_number*sizeof(float));

	//subsample from our image guess
	for (ii=0;ii<use_number;ii++){
		//assigning to subsample
		smallv[ii]=v[use_index[ii]];
		//adding to our counter
		scale=scale+fabs(smallx[ii]);
	}
	scale=scale/fuse_number; //average abs value in our subsample

	for (ii=0;ii<use_number;ii++){
		smallv[ii]=smallv[ii]/scale;
	} //normalizing values to an abs value mean of 1.

	//this seems a bit odd because normal values should be from 0 to 1
	//(since we normalized earlier)
	//but these are mean 1, so it's more like 0-2 or something (depends on dist)

	//subsample B
	//B is bei
	smallB=(float*)malloc(use_number*d*sizeof(float));
	for (jj=0;jj<d;jj++){
		//for each image?
		//but it's not...
		
		for (ii=0;ii<use_number;ii++){
			//for each pixel we want to copy.
			smallB[jj*use_number+ii]=B[jj*n+use_index[ii]];
		}
	}

	//make a copy of tB because pismallB destroys the argument.
	tB=(float*)malloc(use_number*d*sizeof(float));
	for (ii=0;ii<use_number*d;ii++){
		tB[ii]=smallB[ii];
	}

//	fprintf(stderr,"1 in grasta step \n");


//	ticks t1 = getticks();
//	fprintf(stderr,"t1=%g\n",elapsed(t0,t1));

	//get the qr decomposition (specifically, its transpose)
	pismallB=(float*)malloc(use_number*d*sizeof(float));
	pinv_qr_m_big(tB,pismallB,use_number,d);

//update w subsampled, s subsampled, y subsampled
larb_no_orthogonal_alt(pismallB,smallB,use_number,d,smallv,w,s,y,rho,maxiter);


//	ticks t2 = getticks();
//	fprintf(stderr,"t2=%g\n",elapsed(t1,t2));

//[s_t, w, ldual, ~] = sparse_residual_pursuit(U_Omega, y_Omega, OPTS)

/*
	fprintf(stderr,"B[5]=%f\n",B[5]);
	fprintf(stderr,"w[5]=%f\n",w[5]);
*/

	//calculate Γ_1 (stored in g1)

//calculate uw
sgemv("N",&use_number,&d,&one,smallB,&use_number,w,&oneinc,&zero,uw,&oneinc);//uw=B_idx w

	for(jj=0;jj<use_number;jj++){
		g1[jj]=y[jj]+rho*(uw[jj]+s[jj]-smallv[jj]);//-s?   check me!  todo!!!
	}

//calculate Γ_2 (stored in g2)

sgemv("T",&use_number,&d,&one,smallB,&use_number,g1,&oneinc,&zero,g2,&oneinc);//n x use_number g2t=smallB'*g1


sgemv("N",&n,&d,&one,B,&m,g2,&oneinc,&zero,Ug2,&oneinc);//m x n Ug2=B*g2
								//gamma_2 = U0 * UtDual_omega;
//	ticks t3 = getticks();
//	fprintf(stderr,"t3=%g\n",elapsed(t2,t3));


	//3.10:

	//set g to -Ug2
	for(jj=0;jj<m;jj++){
		g[jj] = -Ug2[jj];
	}
	//and add chi omega * g1.
	for(jj=0;jj<use_number;jj++){
		g[use_index[jj]]+=g1[jj];
	}

/*
	gamma = zeros(DIM_M,1);
	gamma(idx) = gamma_1;
	gamma = gamma - gamma_2;
*/



	//calcuate sigma (between 3.11 and 3.12)

	normg=sdot(&n,g,&oneinc,g,&oneinc);
	normg=sqrt(normg);

	normw=sdot(&d,w,&oneinc,w,&oneinc);
	normw=sqrt(normw);

	sigma=normg*normw;
/*

	gamma_norm = norm(gamma);
	w_norm     = norm(w);
	sG = gamma_norm * w_norm;
*/


/*
	cs=(cos(eta*sigma)-1)/(normw*normw);
	ss=sin(eta*sigma)/sigma;

	if ((normw>.01)&&(normw>.01)){
		//reuse g2 as temp:
		for(jj=0;jj<use_number;jj++){
			g2[use_index[jj]]=cs*Uw[jj];
		}
		for(jj=0;jj<m;jj++){
			g2[jj]-=ss*g[jj];//
		}

		for(ii=0;ii<m;ii++){//row
			for(jj=0;jj<n;jj++){//column
				B[jj*m+ii]=B[jj*m+ii]+g2[ii]*w[jj];
			}
		}
	}
*/

//	ticks t4 = getticks();
//	fprintf(stderr,"t4=%g\n",elapsed(t3,t4));

cs=0;
ss=0;



float* alpha;
alpha=(float*)calloc(n,sizeof(float));

//construct the first interior term
if (normw>0){
	cs=(cos(eta*sigma)-1);
	for (ii=0;ii<n;ii++){
		alpha[ii]=uw[ii]/normw^2;
		w[ii]=scale*w[ii];
	}
}

//construct the second interior term
float* beta;
beta=(float*)calloc(d,sizeof(float));
if (normg>0){
	for (ii=0;ii<n;ii++){
		beta[ii]=g[ii]/sigma;
	}
	ss=sin(eta*sigma);
}

sgemv("N",&n,&d,&one,B,&n,alpha,&oneinc,&zero,Uw,&oneinc);//Uw=Bw


//	ticks t5 = getticks();
//	fprintf(stderr,"t5=%g\n",elapsed(t4,t5));


//basically the entire 3.13 calculation at once.
for(ii=0;ii<n;ii++){//row
	for(jj=0;jj<d;jj++){//column
		B[jj*m+ii]=B[jj*m+ii]+(cs*alpha[ii] + ss*beta[ii]) * w[jj];
	}
}



//	ticks t6 = getticks();
//	fprintf(stderr,"t6=%g\n",elapsed(t5,t6));


// Take the gradient step along Grassmannian geodesic.
/*alpha = w/w_norm;
beta  = gamma/gamma_norm;
step = (cos(t)-1)*U0*(alpha*alpha')  - sin(t)*beta*alpha';

U0 = U0 + step;
*/

	free(alpha);
	free(beta);
	free (s);
	free(y);
	free(g1);
	free(uw);
	free(Uw);
	free(Ug2);
	free(g2);
	free(g);
	free(smallv);
	free(smallB);
	free(tB);
	free(pismallB);
}



