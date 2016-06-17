PREFIX=$(HOME)/libraries/install/torch/install/
CFLAGS=-I$(PREFIX)/include/ -I$(PREFIX)/include/THC -I$(PREFIX)/include/TH -I/home/kivan/libraries/install/torch/extra/cunn/lib/THCUNN/
#LDFLAGS_NVCC=-L$(PREFIX)/lib -Xlinker -rpath,$(PREFIX)/lib -lluaT -lTHC -lTH
LDFLAGS_NVCC=-L$(PREFIX)/lib -Xlinker -rpath,$(PREFIX)/lib -lTHC
#LDFLAGS_CPP=-L$(PREFIX)/lib -lluaT -lTH `pkg-config --libs opencv`

all: liblayers.so

#nvcc -arch sm_35 -O3 -DNDEBUG --compiler-options '-fPIC' -o libadcensus.so --shared adcensus.cu $(CFLAGS) $(LDFLAGS_NVCC)
liblayers.so: ClassNLLCriterionWithIgnore.cu
	nvcc -arch sm_52 -O3 -DNDEBUG --compiler-options '-fPIC' -o lib_ClassNLLCriterionWithIgnore.so --shared ClassNLLCriterionWithIgnore.cu $(CFLAGS) $(LDFLAGS_NVCC)

clean:
	rm -f *.so
