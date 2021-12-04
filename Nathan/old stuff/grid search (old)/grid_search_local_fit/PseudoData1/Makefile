OBJS := grid_search_local_fit.c
TARGET_EXEC := grid_search_local_fit
CFLAGS := -std=c99 -Wall -Wextra -Wpedantic -Ofast

DATA_FILE := pseudodata.csv
NUM_SETS := 15

.SILENT: $(TARGET_EXEC) run clean
.PHONY: run clean

$(TARGET_EXEC): *.c *.h Makefile
	gcc $(OBJS) -o $(TARGET_EXEC) $(CFLAGS)

run: $(TARGET_EXEC)
ifdef set
	./$(TARGET_EXEC) $(DATA_FILE) $$set;
else
	for i in {0..$(shell echo $$(($(NUM_SETS)-1)))}; \
	do \
		./$(TARGET_EXEC) $(DATA_FILE) $$i; \
	done
endif

clean:
	rm -f $(TARGET_EXEC) $(TARGET_EXEC)_output.csv
