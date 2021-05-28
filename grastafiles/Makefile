# GRASTAcam demo makefile written by Wisconsin graduate student Jia Xu, © 2012
# Contact Laura Balzano (sunbeam@ece.wisc.edu) with questions.

#MKLROOT = /opt/intel/mkl

MKLROOT = /opt/intel/composer_xe_2011_sp1.11.344/mkl
# opencv include path
IFLAGS = -I/usr/local/include/

# mkl-include path
IFLAGS += -I$(MKLROOT)/include/


# opencv library path
LFLAGS = -L/usr/local/lib -lopencv_core -lopencv_imgproc -lopencv_highgui -lopencv_ml -lopencv_video -lopencv_features2d -lopencv_calib3d -lopencv_objdetect -lopencv_contrib -lopencv_legacy 

# mkl library path
LFLAGS += -L$(MKLROOT)/lib/  -lmkl_intel_lp64 -lmkl_intel_thread  -lmkl_core  -lpthread -lm

LFLAGS += -L/opt/intel/composer_xe_2011_sp1.11.344/compiler/lib/ -liomp5

# blas library  
LFLAGS += -lblas

#all:test 
#all:grastacam
all:grastacam2window


grastacam2window:grastacam_subsample_2window.cpp  grasta.h  larb.h
	g++ -o grastacam2window.o grastacam_subsample_2window.cpp $(IFLAGS) $(LFLAGS)

#grastacam:grastacam_subsample.cpp  grasta.h  larb.h
#	g++ -o grastacam.o grastacam_subsample.cpp $(IFLAGS) $(LFLAGS)


#test:test.cpp  grasta.h  larb.h
#	g++ -o test test.cpp $(IFLAGS) $(LFLAGS)


clean:
	rm *~ *.o grastacam test grastacam2window
