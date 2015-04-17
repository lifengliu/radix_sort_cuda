NVCC = nvcc
NVCCFLAGS=-arch=sm_35
HEADERS=common.h prefixscan.h radixsort.h
CUSOURCES=prefixscan.cu radixsort.cu
CUOBJECTS=$(CUSOURCES:.cu=.o)

all:NVCCFLAGS+=-O3
all:target

debug:clean
deubg:NVCCFLAGS+=-g -G -DDEBUG
debug:target

target:radixsort

radixsort:test_radixsort.cu $(CUOBJECTS) $(HEADERS)
	$(NVCC) $(NVCCFLAGS) -o test_radixsort test_radixsort.cu $(CUOBJECTS)

%.o : %.cu
	$(NVCC) $(NVCCFLAGS) -dc $<

clean:
	rm -f *.o test_radixsort 
