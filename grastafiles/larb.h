//fix frees

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
//this assumes that B has full column rank (we can't get rid of too many rows).

/*
the principle:
B^+ = (B'B)^(-1)B'
so if B = QR
B^+ = (R'Q'QR)^(-1)R'Q'

and because Q is an orthogonal matrix, Q'Q = I.

B^+ = (R'R)^(-1)R'Q
B^+ = R^(-1)R'^(-1)R'Q'
B^+ = R^(-1)Q'
*/

void pinv_qr_m_big(float* B, float* pB,int n,int d){// assumes m>n. destroys B; make a copy.
	float one=1.0f;
	int oneinc=1;
	float zero=0.0f;
	float* r;
	r=(float*)calloc(d*d,sizeof(float));
	float* tri;
	tri=(float*)calloc(d*d,sizeof(float));

	float *tau;
	tau=(float*)malloc(n*sizeof(float));
	int ii,jj;


	float  twork=0;
	int lwork=-1;
	int info;

	//perform qr factorization of B.
	//note that sgeqrf returns a single matrix which has the 
	//upper triangular matrix above the diagonal and 
	//encodes the orthogonal matrix below the diagonal.

	sgeqrf( &n, &d, B, &n, tau, &twork, &lwork, &info);	
	lwork=(int) twork;	
//	printf("\n lwork=%d\n", lwork );		
	float *work;
	work=(float*)malloc(lwork*sizeof(float));

	//I still don't understand why sgeqrf is called twice.
	//Maybe it's necessary for benchmarking?
	sgeqrf(&n, &d, B, &n, tau, work, &lwork, &info );


/*	print_matrix_colmajor(m,n,B);
	printf("\n\n");*/

	//These matrices are stored by stacking the columns.
	//basically ii is x, jj is y.

	//copy the upper portion of the matrix only.
	for (ii=0;ii<d;ii++){//column index
		for (jj=0;jj<(ii+1);jj++){
			r[ii*d+jj]=B[ii*n+jj];
		}
	}


/*
print_matrix_colmajor(n,n,r);
	printf("\n\n");*/

	//properly get the Q for the QR factorization by decoding B.
	sorgqr(&n, &d, &d, B, &n, tau, work, &lwork, &info );

	//so currently, r holds R
	//and B holds Q

	//calculates the inverse of a triangular matrix.
	strtri("U","N",&d,r,&d,&info);
	//so now r holds R^(-1)

	//take the transpose of R^(-1)
	for (ii=0;ii<d;ii++){//column index
		for (jj=0;jj<(ii+1);jj++){
			tri[jj*d+ii]=r[ii*d+jj];
		}
	}
	//so now tri = R^(-1)^T


	//My understanding of this function is that it computes
	//pB = 1 * B * tri + 0*pB
	//which is
	//pB = Q * R^(-1)^T
	//which is the transpose of the pseudoinverse.
	//But in every case where we use it, we take the transpose beforehand.
	//So it cancels
	sgemm("N","N",&n,&d,&d,&one,B,&n,tri,&d,&zero,pB,&n);
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




void larb_orthogonal(float* B,int n,int d,float* v,float* c, float rho,float maxiter){//assumes B is an orthogonal matrix. 
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
	float* y=(float*)calloc(n,sizeof(float));
	float* u=(float*)calloc(n,sizeof(float));
	float* a=(float*)calloc(n,sizeof(float));
	float* junk=(float*)calloc(n,sizeof(float));
	float one=1.0f;
	int oneinc=1;
	float zero=0.0f;

	sgemv("T",&n,&d,&one,B,&n,x,&oneinc,&zero,c,&oneinc);//calculate c
	sgemv("N",&n,&d,&one,B,&n,c,&oneinc,&zero,u,&oneinc);//calculate u=Bc;

	for(jj=0;jj<n;jj++){	
		u[jj]=u[jj]-v[jj];//u=Bc-v;
	}
	//main loop:
	for (ii=0;ii<maxiter;ii++){
		for(jj=0;jj<n;jj++){	
			u[jj]=u[jj]+y[jj];//u=Bc-v+y;
		}
		shrink(u,a,rho,n);//a=shrink(Bc-v+y,rho);
		for(jj=0;jj<n;jj++){	
			junk[jj]=v[jj]+a[jj]-y[jj];//junk=v+a-y
		}

		sgemv("T",&n,&d,&one,B,&n,junk,&oneinc,&zero,c,&oneinc);//calculate c=B'junk
		sgemv("N",&n,&d,&one,B,&n,c,&oneinc,&zero,u,&oneinc);//calculate u=Bc;
		for(jj=0;jj<n;jj++){	
			u[jj]=u[jj]-v[jj];//u=Bc-v;
		}
		for(jj=0;jj<n;jj++){	
			y[jj]=y[jj]-a[jj]+u[jj];//y <-- y-a+u
		}
	}

	//??
//	sgemv(chn,&dp,&d,&one,P+offp,&dp,X+offx,&oneinc,&zero,Z+offz,&oneinc);
}








void larb_no_orthogonal(float* pB, float* B,int n,int d,float* v,float* c, float rho,float maxiter){//does not assume B is an orthogonal matrix; instead, the pseudoinverse pB is passed in.
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
	float* y=(float*)calloc(n,sizeof(float));
	float* u=(float*)calloc(n,sizeof(float));
	float* a=(float*)calloc(n,sizeof(float));
	float* junk=(float*)calloc(n,sizeof(float));
	float one=1.0f;
	int oneinc=1;
	float zero=0.0f;

	sgemv("T",&n,&d,&one,pB,&n,v,&oneinc,&zero,c,&oneinc);//calculate c using c=pBx
	sgemv("N",&n,&d,&one,B,&n,c,&oneinc,&zero,u,&oneinc);//calculate u=Bc;

	for(jj=0;jj<n;jj++){	
		u[jj]=u[jj]-v[jj];//u=Bc-v;
	}
	//main loop:
	for (ii=0;ii<maxiter;ii++){
		for(jj=0;jj<n;jj++){	
			u[jj]=u[jj]+y[jj];//u=Bc-v+y;
		}
		shrink(u,a,rho,n);//a=shrink(Bc-v+y,rho);
		for(jj=0;jj<n;jj++){	
			junk[jj]=v[jj]+a[jj]-y[jj];//junk=v+a-y
		}

		sgemv("T",&n,&d,&one,pB,&n,junk,&oneinc,&zero,c,&oneinc);//calculate c=pB junk
		sgemv("N",&n,&d,&one,B,&n,c,&oneinc,&zero,u,&oneinc);//calculate u=Bc;
		for(jj=0;jj<n;jj++){	
			u[jj]=u[jj]-v[jj];//u=Bc-v;
		}
		for(jj=0;jj<n;jj++){	
			y[jj]=y[jj]-a[jj]+u[jj];//y <-- y-a+u
		}
	}
//	sgemv(chn,&dp,&d,&one,P+offp,&dp,X+offx,&oneinc,&zero,Z+offz,&oneinc);
}














void larb_orthogonal_alt(float* U,int n,int d,float* v,float* w, float* s, float* y,float rho,float maxiter){//assumes U is an orthogonal matrix. USED
	int ii=0,jj=0;
//	float* y=(float*)calloc(n,sizeof(float));
	float* Uw=(float*)calloc(n,sizeof(float));
//	float* a=(float*)calloc(n,sizeof(float));
	float* junk=(float*)calloc(n,sizeof(float));
	float one=1.0f;
	int oneinc=1;
	float zero=0.0f;
	float irho=1/rho;
	//main loop:
	for (ii=0;ii<maxiter;ii++){

		//3.3
		//inside
		for(jj=0;jj<n;jj++){	
			junk[jj]=rho*(v[jj]-s[jj])-y[jj];
		}
	
		//see page 8, equation 3.3. We don't need (U'U)^-1 because we assume orthogonal U.
		//just multiplying by U^T
		sgemv("T",&n,&d,&irho,U,&n,junk,&oneinc,&zero,w,&oneinc);//calculate w=(U'(rho(v-s)-y))/rho


		//3.4:
		//calculate the inside of 3.4: Uw = U*w
		sgemv("N",&n,&d,&one,U,&n,w,&oneinc,&zero,Uw,&oneinc);

		//second stage: prepare junk for input to soft thresh.
		for(jj=0;jj<n;jj++){	
			junk[jj]=v[jj]-Uw[jj]-y[jj];
		}
	
		//soft threshold to get our s.
		shrink(junk,s,1/(1+rho),n);


		//we already calulated u = Uw, so we can just iterate and add.
		for(jj=0;jj<n;jj++){	
			y[jj]=y[jj]+rho*(s[jj]+Uw[jj]-v[jj]);//
		}
/*
		print_matrix_colmajor(n,1,w);
		printf("\n");
*/

	}

free(Uw);
free(junk);

}









void larb_no_orthogonal_alt(float* sU,float* U,int n,int d,float* v,float* w, float* s, float* y,float rho,float maxiter){ // USED
	int ii=0,jj=0;
//	float* y=(float*)calloc(n,sizeof(float));
	float* Uw=(float*)calloc(n,sizeof(float));
//	float* a=(float*)calloc(n,sizeof(float));
	float* junk=(float*)calloc(n,sizeof(float));
	float one=1.0f;
	int oneinc=1;
	float zero=0.0f;
	float irho=1/rho;
	//main loop:
	for (ii=0;ii<maxiter;ii++){


		//generate the inside of 3.3
		for(jj=0;jj<n;jj++){	
			junk[jj]=rho*(v[jj]-s[jj])-y[jj];
		}
		
		//multiply by the Monroe-Penrose pseudoinverse ((U'U)^T)^(-1) * U'
		//so we don't assume orthogonality. This assumes invertibility of (U'U)^T (so we can't cut too many rows...).
		
		//note that sU is actually the transpose of the pseudoinverse of U
		//but we take the transpose again here, so it cancels out.
		
		sgemv("T",&n,&d,&irho,sU,&n,junk,&oneinc,&zero,w,&oneinc);//multiply by the pseudoinverse and divide by rho

		//prep for 3.4
		sgemv("N",&n,&d,&one,U,&n,w,&oneinc,&zero,u,&oneinc);//calculate Uw=U*w;

		//create the inside of the softthresh
		for(jj=0;jj<n;jj++){	
			junk[jj]=v[jj]-Uw[jj]-y[jj];
		}
	
		//use the soft thresh
		shrink(junk,s,1/(1+rho),n);

		//we already have Uw = U*w, so use 3.5 and calculate y.
		for(jj=0;jj<n;jj++){	
			y[jj]=y[jj]+rho*(s[jj]+Uw[jj]-v[jj]);
		}
	}
free(Uw);
free(junk);
}






