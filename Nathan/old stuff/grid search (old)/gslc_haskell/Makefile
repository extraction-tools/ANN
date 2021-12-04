TARGET_EXEC := grid_search_local_fit

C_OBJS := $(TARGET_EXEC).c
HS_OBJS := fit.hs

HSFLAGS := -c -O3 #-Wall -Wextra
CFLAGS := --make -no-hs-main -optc-Ofast

DATA_FILE := dvcs_xs_May-2021_342_sets.csv
NUM_SETS := 342

.SILENT: $(TARGET_EXEC) run clean c_only
.PHONY: run clean

$(TARGET_EXEC): $(HS_OBJS) $(C_OBJS) *.h
	ghc $(HS_OBJS) $(HSFLAGS)
	ghc $(C_OBJS) $(HS_OBJS) -o $(TARGET_EXEC) $(CFLAGS)

c_only:
	# try to compile
	clang $(C_OBJS) -std=c99 -Wall -Wextra -Wpedantic -Weverything

run:
ifdef set
	./$(TARGET_EXEC) $(DATA_FILE) $$set;
else
	for i in {1..$(NUM_SETS)}; \
	do \
		./$(TARGET_EXEC) $(DATA_FILE) $$i; \
	done
endif

clean:
	rm -f $(TARGET_EXEC) $(TARGET_EXEC)_output.csv *.o *.hi *_stub.h
