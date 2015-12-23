CC = g++
LDFLAGS = -lgsl -lm -lgslcblas


LSOURCE = main.cpp corpus.cpp maslda.cpp utils.cpp opt.cpp
LHEADER = corpus.h maslda.h utils.h opt.h settings.h

maslda: $(LSOURCE) $(HEADER)
	  $(CC) $(LSOURCE) -o $@ $(LDFLAGS)

clean:
	-rm -f *.o maslda
