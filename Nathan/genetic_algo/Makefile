NUM_SETS := 124

all: *.c *.h
	clang genetic_algo.c -o genetic_algo -Weverything -std=c99 -Ofast

run:
ifdef set
	./genetic_algo $$set;
else
	for (( i=1; i<=$(NUM_SETS); i+=1 )); \
	do \
		./genetic_algo $$i; \
	done
endif

clean:
	rm -rf .vscode
	rm -rf *.dSYM/
#rm -f ./genetic_algo