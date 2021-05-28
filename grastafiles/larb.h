//fix frees!

void print_matrix_colmajor(int numrows,int numcols,float* M){
    int ii,jj;
    for (jj=0;jj<numrows;jj++){
        for (ii=0;ii<numcols;ii++){
           
	 	fprintf(stderr,"%-10f",M[ii*numrows+jj]);
	
		
        }
        fprintf(stderr,"\n");
    }
}





void print_matrix_colmajor(int numrows,int numcols,int* M){
    int ii,jj;
    for (jj=0;jj<numrows;jj++){
        for (ii=0;ii<numcols;ii++){
           
	 	fprintf(stderr,"%-10d",M[ii*numrows+jj]);
	
		
        }
        fprintf(stderr,"\n");
    }
}

//returns a monroe-penrose pseudoinverse via qr factorization.
void pinv_qr_m_big(float* B, float* pB,int m,int n){// assumes m>n. destroys B; make a copy.
	float one=1.0f;
	int oneinc=1;
	float zero=0.0f;
	float* r;
	r=(float*)calloc(n*n,sizeof(float));
	float* tri;
	tri=(float*)calloc(n*n,sizeof(float));

	float *tau;
	tau=(float*)malloc(m*sizeof(float));
	int ii,jj;


	float  twork=0;
	int lwork=-1;
	int info;

	sgeqrf( &m, &n, B, &m, tau, &twork, &lwork, &info);	
	lwork=(int) twork;	
//	printf("\n lwork=%d\n", lwork );		
	float *work;
	work=(float*)malloc(lwork*sizeof(float));

	sgeqrf(&m, &n, B, &m, tau, work, &lwork, &info );


/*	print_matrix_colmajor(m,n,B);
	printf("\n\n");*/

	for (ii=0;ii<n;ii++){//column index
		for (jj=0;jj<(ii+1);jj++){
			r[ii*n+jj]=B[ii*m+jj];
		}
	}


/*
print_matrix_colmajor(n,n,r);
	printf("\n\n");*/

	sorgqr(&m, &n, &n, B, &m, tau, work, &lwork, &info );


	strtri("U","N",&n,r,&n,&info);

	for (ii=0;ii<n;ii++){//column index
		for (jj=0;jj<(ii+1);jj++){
			tri[jj*n+ii]=r[ii*n+jj];
		}
	}

	
	sgemm("N","N",&m,&n,&n,&one,B,&m,tri,&n,&zero,pB,&m);
	free(work);
	free(tri);
	free(r);
	free(tau);
}




void shrink(float* v,float* s,float rho,int N){
	int ii;
	float t=0;
	for(ii=0;ii<N;ii++){
		t=v[ii];
		s[ii]=(t-rho*((t > 0) - (t < 0)))*(fabs(t)>rho);
	}
}




void larb_orthogonal(float* B,int m,int n,float* v,float* c, float rho,float maxiter){//assumes B is an orthogonal matrix. 
/*
c=B'*v;
u=B*c-v;
y=0;
for k=1:maxiter
    a=shrink(B*c-v+y,rho);
    c=B'*(v+a-y);
    y=y-a+(B*c-v);
end

*/
	int ii=0,jj=0;
	float* y=(float*)calloc(m,sizeof(float));
	float* u=(float*)calloc(m,sizeof(float));
	float* a=(float*)calloc(m,sizeof(float));
	float* junk=(float*)calloc(m,sizeof(float));
	float one=1.0f;
	int oneinc=1;
	float zero=0.0f;

	sgemv("T",&m,&n,&one,B,&m,x,&oneinc,&zero,c,&oneinc);//calculate c
	sgemv("N",&m,&n,&one,B,&m,c,&oneinc,&zero,u,&oneinc);//calculate u=Bc;

	for(jj=0;jj<m;jj++){	
		u[jj]=u[jj]-v[jj];//u=Bc-v;
	}
	//main loop:
	for (ii=0;ii<maxiter;ii++){
		for(jj=0;jj<m;jj++){	
			u[jj]=u[jj]+y[jj];//u=Bc-v+y;
		}
		shrink(u,a,rho,m);//a=shrink(Bc-v+y,rho);
		for(jj=0;jj<m;jj++){	
			junk[jj]=v[jj]+a[jj]-y[jj];//junk=v+a-y
		}

		sgemv("T",&m,&n,&one,B,&m,junk,&oneinc,&zero,c,&oneinc);//calculate c=B'junk
		sgemv("N",&m,&n,&one,B,&m,c,&oneinc,&zero,u,&oneinc);//calculate u=Bc;
		for(jj=0;jj<m;jj++){	
			u[jj]=u[jj]-v[jj];//u=Bc-v;
		}
		for(jj=0;jj<m;jj++){	
			y[jj]=y[jj]-a[jj]+u[jj];//y <-- y-a+u
		}
	}

	//??
//	sgemv(chn,&dp,&d,&one,P+offp,&dp,X+offx,&oneinc,&zero,Z+offz,&oneinc);
}








void larb_no_orthogonal(float* pB, float* B,int m,int n,float* v,float* c, float rho,float maxiter){//does not assume B is an orthogonal matrix; instead, the pseudoinverse pB is passed in.
/*
c=B'*v;
u=B*c-v;
y=0;
for k=1:maxiter
    a=shrink(B*c-x+y,rho);
    c=B'*(x+a-y);
    y=y-a+(B*c-x);
end

*/




	int ii=0,jj=0;
	float* y=(float*)calloc(m,sizeof(float));
	float* u=(float*)calloc(m,sizeof(float));
	float* a=(float*)calloc(m,sizeof(float));
	float* junk=(float*)calloc(m,sizeof(float));
	float one=1.0f;
	int oneinc=1;
	float zero=0.0f;

	sgemv("T",&m,&n,&one,pB,&m,v,&oneinc,&zero,c,&oneinc);//calculate c using c=pBx
	sgemv("N",&m,&n,&one,B,&m,c,&oneinc,&zero,u,&oneinc);//calculate u=Bc;

	for(jj=0;jj<m;jj++){	
		u[jj]=u[jj]-v[jj];//u=Bc-v;
	}
	//main loop:
	for (ii=0;ii<maxiter;ii++){
		for(jj=0;jj<m;jj++){	
			u[jj]=u[jj]+y[jj];//u=Bc-v+y;
		}
		shrink(u,a,rho,m);//a=shrink(Bc-v+y,rho);
		for(jj=0;jj<m;jj++){	
			junk[jj]=v[jj]+a[jj]-y[jj];//junk=v+a-y
		}

		sgemv("T",&m,&n,&one,pB,&m,junk,&oneinc,&zero,c,&oneinc);//calculate c=pB junk
		sgemv("N",&m,&n,&one,B,&m,c,&oneinc,&zero,u,&oneinc);//calculate u=Bc;
		for(jj=0;jj<m;jj++){	
			u[jj]=u[jj]-v[jj];//u=Bc-v;
		}
		for(jj=0;jj<m;jj++){	
			y[jj]=y[jj]-a[jj]+u[jj];//y <-- y-a+u
		}
	}
//	sgemv(chn,&dp,&d,&one,P+offp,&dp,X+offx,&oneinc,&zero,Z+offz,&oneinc);
}














void larb_orthogonal_alt(float* B,int m,int n,float* v,float* w, float* s, float* y,float rho,float maxiter){//assumes B is an orthogonal matrix. 
	int ii=0,jj=0;
//	float* y=(float*)calloc(m,sizeof(float));
	float* u=(float*)calloc(m,sizeof(float));
//	float* a=(float*)calloc(m,sizeof(float));
	float* junk=(float*)calloc(m,sizeof(float));
	float one=1.0f;
	int oneinc=1;
	float zero=0.0f;
	float irho=1/rho;
	//main loop:
	for (ii=0;ii<maxiter;ii++){

		//3.3
		//inside
		for(jj=0;jj<m;jj++){	
			junk[jj]=rho*(v[jj]-s[jj])-y[jj];
		}
	
		//see page 8, equation 3.3. We don't need (B'B)^-1 because we assume orthogonal B.
		//just multiplying by B^T
		sgemv("T",&m,&n,&irho,B,&m,junk,&oneinc,&zero,w,&oneinc);//calculate w=(B'(rho(v-s)-y))/rho


		//3.4:
		//calculate the inside of 3.4: u = Bw
		sgemv("N",&m,&n,&one,B,&m,w,&oneinc,&zero,u,&oneinc);

		//second stage: prepare junk for input to soft thresh.
		for(jj=0;jj<m;jj++){	
			junk[jj]=v[jj]-u[jj]-y[jj];
		}
	
		//soft threshold to get our s.
		shrink(junk,s,1/(1+rho),m);


		//we already calulated u = Bw, so we can just iterate and add.
		for(jj=0;jj<m;jj++){	
			y[jj]=y[jj]+rho*(s[jj]+u[jj]-v[jj]);//
		}
/*
		print_matrix_colmajor(n,1,w);
		printf("\n");
*/

	}

free(u);
free(junk);

}









void larb_no_orthogonal_alt(float* sB,float* B,int m,int n,float* v,float* w, float* s, float* y,float rho,float maxiter){ 
	int ii=0,jj=0;
//	float* y=(float*)calloc(m,sizeof(float));
	float* u=(float*)calloc(m,sizeof(float));
//	float* a=(float*)calloc(m,sizeof(float));
	float* junk=(float*)calloc(m,sizeof(float));
	float one=1.0f;
	int oneinc=1;
	float zero=0.0f;
	float irho=1/rho;
	//main loop:
	for (ii=0;ii<maxiter;ii++){


		//generate the inside of 3.3
		for(jj=0;jj<m;jj++){	
			junk[jj]=rho*(v[jj]-s[jj])-y[jj];
		}
		
		//multiply by the pseudoinverse ((B'B)^T)^(-1) * B'
		//so we don't assume orthogonality. This assumes invertibility of (B'B)^T.
		sgemv("T",&m,&n,&irho,sB,&m,junk,&oneinc,&zero,w,&oneinc);//calculate w=(B'(rho(v-s)-y))/rho

		//prep for 3.4
		sgemv("N",&m,&n,&one,B,&m,w,&oneinc,&zero,u,&oneinc);//calculate u=Bw;

		//create the inside of the softthresh
		for(jj=0;jj<m;jj++){	
			junk[jj]=v[jj]-u[jj]-y[jj];
		}
	
		//use the soft thresh
		shrink(junk,s,1/(1+rho),m);

		//we already have u = Bw, so use 3.5 and calculate y.
		for(jj=0;jj<m;jj++){	
			y[jj]=y[jj]+rho*(s[jj]+u[jj]-v[jj]);//
		}
	}
free(u);
free(junk);
}






