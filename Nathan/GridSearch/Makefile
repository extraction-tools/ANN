# GRID SEARCH BUILD SCRIPT
# Nathan Snyder (nbs9vy)

# Names
LOCAL_FIT_NAME := LocalFit

LOCAL_FIT_BOOTSTRAPPING_NAME := $(LOCAL_FIT_NAME)/Bootstrapping
LOCAL_FIT_PARALLEL_NAME := $(LOCAL_FIT_NAME)/Parallel
LOCAL_FIT_SEQUENTIAL_NAME := $(LOCAL_FIT_NAME)/Sequential
LOCAL_FIT_NOREPLICAS_NAME := $(LOCAL_FIT_NAME)/NoReplicas

CLEANER_NAME := Cleaner
NARROWER_NAME := Narrower
GLOBAL_FIT_NAME := GlobalFit

LOCAL_FIT_PATH := src/$(LOCAL_FIT_NAME)
LOCAL_FIT_BOOTSTRAPPING_PATH := src/$(LOCAL_FIT_NAME)/Bootstrapping
LOCAL_FIT_PARALLEL_PATH := src/$(LOCAL_FIT_NAME)/Parallel
LOCAL_FIT_SEQUENTIAL_PATH := src/$(LOCAL_FIT_NAME)/Sequential
LOCAL_FIT_NOREPLICAS_PATH := src/$(LOCAL_FIT_NAME)/NoReplicas

# Dependencies
LOCAL_FIT_BOOTSTRAPPING_DEPENDENCIES := src/$(LOCAL_FIT_BOOTSTRAPPING_NAME)/*.go
LOCAL_FIT_PARALLEL_DEPENDENCIES := src/$(LOCAL_FIT_PARALLEL_NAME)/*.go
LOCAL_FIT_SEQUENTIAL_DEPENDENCIES := src/$(LOCAL_FIT_SEQUENTIAL_NAME)/*.c src/$(LOCAL_FIT_SEQUENTIAL_NAME)/*.h
LOCAL_FIT_NOREPLICAS_DEPENDENCIES := src/$(LOCAL_FIT_NOREPLICAS_NAME)/*.c src/$(LOCAL_FIT_NOREPLICAS_NAME)/*.h

CLEANER_DEPENDENCIES := 
NARROWER_DEPENDENCIES :=
GLOBAL_FIT_DEPENDENCIES := 

# Target executables
LOCAL_FIT_TARGET_EXEC_FROM_PATH := ../../../bin/$(LOCAL_FIT_NAME)

LOCAL_FIT_BOOTSTRAPPING_EXEC_FROM_PATH := $(LOCAL_FIT_TARGET_EXEC_FROM_PATH)Bootstrapping
LOCAL_FIT_PARALLEL_EXEC_FROM_PATH := $(LOCAL_FIT_TARGET_EXEC_FROM_PATH)Parallel
LOCAL_FIT_SEQUENTIAL_EXEC_FROM_PATH := $(LOCAL_FIT_TARGET_EXEC_FROM_PATH)Sequential
LOCAL_FIT_NOREPLICAS_EXEC_FROM_PATH := $(LOCAL_FIT_TARGET_EXEC_FROM_PATH)NoReplicas

LOCAL_FIT_BOOTSTRAPPING_EXEC_FROM_ROOT := ./bin/LocalFitBootstrapping
LOCAL_FIT_PARALLEL_EXEC_FROM_ROOT := ./bin/LocalFitParallel
LOCAL_FIT_SEQUENTIAL_EXEC_FROM_ROOT := ./bin/LocalFitSequential
LOCAL_FIT_NOREPLICAS_EXEC_FROM_ROOT := ./bin/LocalFitNoReplicas

CLEANER_TARGET_EXEC := bin/$(CLEANER_NAME)
NARROWER_TARGET_EXEC := bin/$(NARROWER_NAME)
GLOBAL_FIT_TARGET_EXEC := bin/$(GLOBAL_FIT_NAME)

# Objects
# No local fit objects
#LOCAL_FIT_OBJS := 
CLEANER_OBJS := src/$(CLEANER_NAME)/Cleaner.c
NARROWER_OBJS := src/$(NARROWER_NAME)/Narrower.c
GLOBAL_FIT_OBJS := src/$(GLOBAL_FIT_NAME)/GlobalFit.c

# Flags (for all builds)
CFLAGS := -std=c99 -Weverything -Ofast
GOFLAGS := 

# Location of (pseudo)data
# Pseudodata 1: lib/dvcs_xs_newsets_withCFFs_2.csv
# Pseudodata 2: lib/dvcs_xs_May-2021_342_sets.csv
# Pseudodata 3: lib/dvcs_bkm2002_June2021_4pars.csv
PSEUDODATA_FILE := lib/dvcs_xs_newsets_withCFFs_2.csv

# Input files
LOCAL_FIT_INPUT_FILE := $(PSEUDODATA_FILE)
CLEANER_INPUT_FILE := 
NARROWER_INPUT_FILE := 
GLOBAL_FIT_INPUT_FILE := 

# Output files
LOCAL_FIT_OUTPUT_FILE := results/$(LOCAL_FIT_NAME)/$(LOCAL_FIT_NAME)Output.csv
CLEANER_OUTPUT_FILE := results/$(CLEANER_NAME)/$(CLEANER_NAME)Output.csv
NARROWER_OUTPUT_FILE := results/$(NARROWER_NAME)/$(NARROWER_NAME)Output.csv
GLOBAL_FIT_OUTPUT_FILE := results/$(GLOBAL_FIT_NAME)/$(GLOBAL_FIT_NAME)Output.csv

# Number of sets in data
NUM_SETS := 100


.PHONY: all clean
.ONESHELL:
.SILENT:

SHELL=/bin/bash



all:
	make local_fits
	make cleaner
	make narrower
	make global_fit

local_fits: 
	make --no-print-directory local_fit_parallel
	make --no-print-directory local_fit_sequential
	make --no-print-directory local_fit_bootstrapping

local_fit_bootstrapping: $(LOCAL_FIT_BOOTSTRAPPING_DEPENDENCIES)
	echo "Making $(LOCAL_FIT_BOOTSTRAPPING_NAME)..."; \
	cd $(LOCAL_FIT_BOOTSTRAPPING_PATH); \
	go build -o $(LOCAL_FIT_BOOTSTRAPPING_EXEC_FROM_PATH); \
	chmod +rwx $(LOCAL_FIT_BOOTSTRAPPING_EXEC_FROM_PATH); \
	echo "Done."

local_fit_parallel: $(LOCAL_FIT_PARALLEL_DEPENDENCIES)
	echo "Making $(LOCAL_FIT_PARALLEL_NAME)..."; \
	cd $(LOCAL_FIT_PARALLEL_PATH); \
	go build -o $(LOCAL_FIT_PARALLEL_EXEC_FROM_PATH); \
	chmod +rwx $(LOCAL_FIT_PARALLEL_EXEC_FROM_PATH); \
	echo "Done."

local_fit_sequential: $(LOCAL_FIT_SEQUENTIAL_DEPENDENCIES)
	echo "Making $(LOCAL_FIT_SEQUENTIAL_NAME)..."; \
	cd $(LOCAL_FIT_SEQUENTIAL_PATH); \
	clang *.c -o $(LOCAL_FIT_SEQUENTIAL_EXEC_FROM_PATH) $(CFLAGS); \
	chmod +rwx $(LOCAL_FIT_SEQUENTIAL_EXEC_FROM_PATH); \
	echo "Done."

local_fit_noreplicas: $(LOCAL_FIT_NOREPLICAS_DEPENDENCIES)
	echo "Making $(LOCAL_FIT_NOREPLICAS_NAME)..."; \
	cd $(LOCAL_FIT_NOREPLICAS_PATH); \
	clang *.c -o $(LOCAL_FIT_NOREPLICAS_EXEC_FROM_PATH) $(CFLAGS); \
	chmod +rwx $(LOCAL_FIT_NOREPLICAS_EXEC_FROM_PATH); \
	echo "Done."



cleaner: $(CLEANER_OBJS) $(CLEANER_DEPENDENCIES)
	clang $(CLEANER_OBJS) -o $(CLEANER_TARGET_EXEC) $(CFLAGS)

narrower: $(NARROWER_OBJS) $(NARROWER_DEPENDENCIES)
	clang $(NARROWER_OBJS) -o $(NARROWER_TARGET_EXEC) $(CFLAGS)

global_fit: $(GLOBAL_FIT_OBJS) $(GLOBAL_FIT_DEPENDENCIES)
	clang $(GLOBAL_FIT_OBJS) -o $(GLOBAL_FIT_TARGET_EXEC) $(CFLAGS)


run_local_fit_sequential:
ifdef set
	$(LOCAL_FIT_SEQUENTIAL_EXEC_FROM_ROOT) $$set;
else
	for (( i=4; i<=14; i+=1 )); \
	do \
		$(LOCAL_FIT_SEQUENTIAL_EXEC_FROM_ROOT) $$i; \
	done
endif

run_local_fit_noreplicas:
ifdef set
	$(LOCAL_FIT_NOREPLICAS_EXEC_FROM_ROOT) $(PSEUDODATA_FILE) $$set;
else
	for (( i=1; i<=$(NUM_SETS); i+=1 )); \
	do \
		$(LOCAL_FIT_NOREPLICAS_EXEC_FROM_ROOT) $(PSEUDODATA_FILE) $$i; \
	done
endif

run_local_fit_bootstrapping:
ifdef set
	$(LOCAL_FIT_BOOTSTRAPPING_EXEC_FROM_ROOT) $$set;
else
	for (( i=0; i<=0; i+=1 )); \
	do \
		echo "Set $$i:"; \
		$(LOCAL_FIT_BOOTSTRAPPING_EXEC_FROM_ROOT) $$i; \
	done
endif

run_global_fit:
	./$(GLOBAL_FIT_TARGET_EXEC) $(GLOBAL_FIT_INPUT_FILE)

clean:
	rm -rf bin/*
