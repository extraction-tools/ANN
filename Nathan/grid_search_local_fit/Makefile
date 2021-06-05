OBJS := grid_search_local_fit.c
TARGET_EXEC := grid_search_local_fit
CFLAGS := -std=c99 -Wall -Wextra -Wpedantic -Weverything -Ofast

DATA_FILE := dvcs_xs_May-2021_342_sets.csv
NUM_SETS := 342

.SILENT: $(TARGET_EXEC) run clean
.PHONY: run clean

$(TARGET_EXEC): *.c *.h Makefile
	clang $(OBJS) -o $(TARGET_EXEC) $(CFLAGS)

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
	rm -f $(TARGET_EXEC) $(TARGET_EXEC)_output.csv
