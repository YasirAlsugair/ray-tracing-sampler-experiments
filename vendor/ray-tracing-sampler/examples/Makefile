LIB_FLAGS=
CFLAGS=$(LIB_FLAGS) -lm -m64 -D_LARGEFILE_SOURCE -D_LARGEFILE64_SOURCE -D_FILE_OFFSET_BITS=64 -D_DEFAULT_SOURCE -D_POSIX_SOURCE -D_POSIX_C_SOURCE=200809L -D_DARWIN_C_SOURCE  -Wall -fno-math-errno -std=c99
OPT_FLAGS=-O3
DEBUG_FLAGS=-g -Wall -O
PROF_FLAGS=-g -pg -ftest-coverage -fprofile-arcs

all:
	@make reg EXTRA_FLAGS="$(OPT_FLAGS)"

debug:
	@make reg EXTRA_FLAGS="$(DEBUG_FLAGS)"

prof:
	@make reg EXTRA_FLAGS="$(PROF_FLAGS)"

reg:
	$(CC) gaussian.c ../raytrace.c  $(CFLAGS) $(EXTRA_FLAGS) -o gaussian

clean:
	rm -f gaussian




