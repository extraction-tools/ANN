LOCAL_FIT_DEPENDENCIES := TVA1_UU.h LorentzVector.h
GLOBAL_FIT_DEPENDENCIES := 

LOCAL_FIT_TARGET_EXEC := local_fit
LOCAL_FIT_RESULT_CLEANER_EXEC := local_fit_result_cleaner
GLOBAL_FIT_TARGET_EXEC := global_fit

LOCAL_FIT_OBJS := $(LOCAL_FIT_TARGET_EXEC).c
LOCAL_FIT_RESULT_CLEANER_OBJS := $(LOCAL_FIT_RESULT_CLEANER_EXEC).c
GLOBAL_FIT_OBJS := $(GLOBAL_FIT_TARGET_EXEC).c

CFLAGS := -std=c99 -Wall -Wextra -Wpedantic -Weverything -Ofast

LOCAL_FIT_INPUT_FILE := dvcs_xs_May-2021_342_sets.csv
GLOBAL_FIT_INPUT_FILE := local_fit_output.csv

LOCAL_FIT_OUTPUT_FILE := local_fit_output.csv
GLOBAL_FIT_OUTPUT_FILE := 

NUM_SETS := 342

.SILENT: local_fit global_fit run_local_fit run_global_fit clean
.PHONY: all local_fit global_fit run run_local_fit run_global_fit clean

SHELL=/bin/bash

all: local_fit global_fit

local_fit: $(LOCAL_FIT_OBJS) $(LOCAL_FIT_DEPENDENCIES)
	clang $(LOCAL_FIT_OBJS) -o $(LOCAL_FIT_TARGET_EXEC) $(CFLAGS)

global_fit: $(GLOBAL_FIT_OBJS) $(GLOBAL_FIT_DEPENDENCIES)
	clang $(GLOBAL_FIT_OBJS) -o $(GLOBAL_FIT_TARGET_EXEC) $(CFLAGS)

run: run_local_fit run_global_fit

run_local_fit:
ifdef set
	./$(LOCAL_FIT_TARGET_EXEC) $(LOCAL_FIT_INPUT_FILE) $$set;
else
	for (( i=1; i<=$(NUM_SETS); i+=1 )); \
	do \
		./$(LOCAL_FIT_TARGET_EXEC) $(LOCAL_FIT_INPUT_FILE) $$i; \
	done
endif

run_global_fit:
	./$(GLOBAL_FIT_TARGET_EXEC) $(GLOBAL_FIT_INPUT_FILE)

clean:
	rm -f $(LOCAL_FIT_TARGET_EXEC) $(GLOBAL_FIT_TARGET_EXEC) *.o
	rm -rf *.dSYM/


	for (( i=1; i<=$(NUM_SETS); i+=4 )); \
	do \
		./$(LOCAL_FIT_TARGET_EXEC) $(LOCAL_FIT_INPUT_FILE) $$i & \
		./$(LOCAL_FIT_TARGET_EXEC) $(LOCAL_FIT_INPUT_FILE) $$(( $$i + 1 )) & \
		./$(LOCAL_FIT_TARGET_EXEC) $(LOCAL_FIT_INPUT_FILE) $$(( $$i + 2 )) & \
		./$(LOCAL_FIT_TARGET_EXEC) $(LOCAL_FIT_INPUT_FILE) $$(( $$i + 3 )) & \
		wait; \
	done