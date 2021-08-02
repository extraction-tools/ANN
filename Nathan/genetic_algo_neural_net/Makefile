NUM_SETS := 124

all: *.c *.h
	clang genetic_algo_neural_net.c -o genetic_algo_neural_net -Weverything -std=c99 -O2

run:
	./genetic_algo_neural_net

clean:
	rm -rf .vscode
	rm -rf *.dSYM/
#rm -f ./genetic_algo_neural_net