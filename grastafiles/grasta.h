//#include "cycle.h"



void grasta_step (float* B, float* v, float* w, int m, int n, float eta,float rho, int maxiter){

	float one=1.0f;
	int oneinc=1;
	float zero=0.0f;
	int ii,jj;
	//float* w=(float*)malloc(n*sizeof(float));
	float* s=(float*)calloc(m,sizeof(float));
	float* y=(float*)calloc(m,sizeof(float));
	float* g1=(float*)malloc(m*sizeof(float));
	float* uw=(float*)malloc(m*sizeof(float));
	float* g2t=(float*)malloc(n*sizeof(float));
	float* g2=(float*)malloc(m*sizeof(float));
	float* g=(float*)malloc(m*sizeof(float));
	float sigma;
	float normg;
	float normw;
	float cs;
	float ss;


	//update w,s,y
	larb_orthogonal_alt(B,m,n,v,w,s,y,rho,maxiter);


	//calculate Γ_1 (stored in g1)

	//uw = Uw
	sgemv("N",&m,&n,&one,B,&m,w,&oneinc,&zero,uw,&oneinc); 
	for(jj=0;jj<m;jj++){
		g1[jj]=y[jj]+rho*(uw[jj]+s[jj]-v[jj]);
	}

	//calculate Γ_2 (stored in g2t)
	sgemv("T",&m,&n,&one,B,&m,g1,&oneinc,&zero,g2t,&oneinc);


	//calculate Γ. 

	//g2 = U(Γ_2)
	sgemv("N",&m,&n,&one,B,&m,g2t,&oneinc,&zero,g2,&oneinc);

	//final calculation of Γ (stored in g)
	for(jj=0;jj<m;jj++){
		g[jj]=g1[jj]-g2[jj];
	}

	//calcuate sigma (between 3.11 and 3.12)
	normg=sdot(&m,g,&oneinc,g,&oneinc);
	normg=sqrt(normg);

	normw=sdot(&n,w,&oneinc,w,&oneinc);
	normw=sqrt(normw);

	sigma=normg*normw;


	//todo: how do you explain this 
	cs=(cos(eta*sigma)-1)/(normw*normw);
	ss=sin(eta*sigma)/sigma;

	//reuse g1 as temp:
	for(jj=0;jj<m;jj++){
		g1[jj]=cs*uw[jj]-ss*g[jj];
	}

	for(ii=0;ii<m;ii++){//row
		for(jj=0;jj<n;jj++){//column
			B[jj*m+ii]=B[jj*m+ii]+g1[ii]*w[jj];
		}
	}

	free (s);
	free(y);
	free(g1);
	free(uw);
	free(g2t);
	free(g2);
	free(g);
}























void grasta_step_subsample(float* B, float* v, float* w, int m, int n, float eta,float rho, int maxiter, int* use_index, int use_number){

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
	float* Uw=(float*)malloc(m*sizeof(float));
	float* g2t=(float*)malloc(n*sizeof(float));
	float* g2=(float*)malloc(m*sizeof(float));
	float* g=(float*)malloc(m*sizeof(float));
	float sigma;
	float normg;
	float normw;
	float cs;
	float ss;
	float scale=0;
	float fuse_number=(float)use_number;


//	ticks t0 = getticks();

//    fprintf(stderr,"0 in grasta step \n");
	smallv=(float*)malloc(use_number*sizeof(float));
	for (ii=0;ii<use_number;ii++){
		smallv[ii]=v[use_index[ii]];
		scale=scale+fabs(smallx[ii]);
	}
	scale=scale/fuse_number;
	for (ii=0;ii<use_number;ii++){
		smallv[ii]=smallv[ii]/scale;
	}

	smallB=(float*)malloc(use_number*n*sizeof(float));
	for (jj=0;jj<n;jj++){
		for (ii=0;ii<use_number;ii++){
			smallB[jj*use_number+ii]=B[jj*m+use_index[ii]];
		}
	}

	tB=(float*)malloc(use_number*n*sizeof(float));
	for (ii=0;ii<use_number*n;ii++){
		tB[ii]=smallB[ii];
	}

//	fprintf(stderr,"1 in grasta step \n");


//	ticks t1 = getticks();
//	fprintf(stderr,"t1=%g\n",elapsed(t0,t1));


	pismallB=(float*)malloc(use_number*n*sizeof(float));
	pinv_qr_m_big(tB,pismallB,use_number,n);


larb_no_orthogonal_alt(pismallB,smallB,use_number,n,smallv,w,s,y,rho,maxiter);


//	ticks t2 = getticks();
//	fprintf(stderr,"t2=%g\n",elapsed(t1,t2));

//[s_t, w, ldual, ~] = sparse_residual_pursuit(U_Omega, y_Omega, OPTS)

/*
	fprintf(stderr,"B[5]=%f\n",B[5]);
	fprintf(stderr,"w[5]=%f\n",w[5]);
*/

sgemv("N",&use_number,&n,&one,smallB,&use_number,w,&oneinc,&zero,uw,&oneinc);//uw=B_idx w

	for(jj=0;jj<use_number;jj++){
		g1[jj]=y[jj]+rho*(uw[jj]+s[jj]-smallv[jj]);//-s?   check me!  todo!!!
	}
//gamma_1 = ldual + OPTS.RHO*(U_Omega*w + s_t - y_Omega);



sgemv("T",&use_number,&n,&one,smallB,&use_number,g1,&oneinc,&zero,g2t,&oneinc);//n x use_number g2t=smallB'*g1

//UtDual_omega = U_Omega' * gamma_1;
	sgemv("N",&m,&n,&one,B,&m,g2t,&oneinc,&zero,g2,&oneinc);//m x n g2=B*g2t
								//gamma_2 = U0 * UtDual_omega;
//	ticks t3 = getticks();
//	fprintf(stderr,"t3=%g\n",elapsed(t2,t3));


	for(jj=0;jj<m;jj++){
		g[jj]=-g2[jj];
	}
	for(jj=0;jj<use_number;jj++){
		g[use_index[jj]]+=g1[jj];
	}

/*
	gamma = zeros(DIM_M,1);
	gamma(idx) = gamma_1;
	gamma = gamma - gamma_2;
*/





	normg=sdot(&m,g,&oneinc,g,&oneinc);
	normg=sqrt(normg);

	normw=sdot(&n,w,&oneinc,w,&oneinc);
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
alpha=(float*)calloc(m,sizeof(float));

if (normw>0){
	cs=(cos(eta*sigma)-1);
	for (ii=0;ii<n;ii++){
		alpha[ii]=w[ii]/normw;
		w[ii]=scale*w[ii];
	}
}

float* beta;
beta=(float*)calloc(m,sizeof(float));
if (normg>0){
	for (ii=0;ii<m;ii++){
		beta[ii]=g[ii]/normg;
	}
	ss=sin(eta*sigma);
}

sgemv("N",&m,&n,&one,B,&m,alpha,&oneinc,&zero,Uw,&oneinc);//Uw=Bw


//	ticks t5 = getticks();
//	fprintf(stderr,"t5=%g\n",elapsed(t4,t5));


for(ii=0;ii<m;ii++){//row
	for(jj=0;jj<n;jj++){//column

B[jj*m+ii]=B[jj*m+ii]+cs*Uw[ii]*alpha[jj]-ss*beta[ii]*alpha[jj];
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
	free(g2t);
	free(g2);
	free(g);
	free(smallv);
	free(smallB);
	free(tB);
	free(pismallB);
}



