FLAGS= -DDEBUG -arch=sm_60
LIBS= -lm
ALWAYS_REBUILD=makefile
CC=nvcc

nbody: nbody.o compute.o
	$(CC) $(FLAGS) $^ -o $@ $(LIBS)
nbody.o: nbody.c planets.h config.h vector.h $(ALWAYS_REBUILD)
	$(CC) $(FLAGS) -c $< 
compute.o: compute.cu config.h vector.h $(ALWAYS_REBUILD)
	$(CC) $(FLAGS) -c $< 
clean:
	rm -f *.o nbody 
