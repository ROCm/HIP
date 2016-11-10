HCC_HOME?=/opt/rocm/hcc
HCC = $(HCC_HOME)/bin/hcc

OPT=-O2
HCC_CFLAGS= `$(HCC_HOME)/bin/hcc-config --cxxflags` ${OPT}
HCC_LDFLAGS= `$(HCC_HOME)/bin/hcc-config --ldflags` ${OPT}

CPPAMP_CFLAGS= `$(HCC_HOME)/bin/clamp-config --cxxflags`
CPPAMP_LDFLAGS= `$(HCC_HOME)/bin/clamp-config --ldflags`

HIP_PATH?= $(wildcard /opt/rocm/hip)
ifeq (,$(HIP_PATH))
	HIP_PATH=../../..
endif
HIPCC=$(HIP_PATH)/bin/hipcc
HIP_PLATFORM=$(shell $(HIP_PATH)/bin/hipconfig --platform)

ifneq (${HIP_PLATFORM}, hcc)
$(error hcc_dialects requires hcc compiler and only runs on hcc platform)
endif


TARGETS=vadd_hc_arrayview vadd_hc_array vadd_hc_am vadd_amp_arrayview  vadd_hip  

all: $(TARGETS)

clean: 
	rm -f $(TARGETS) *.o

run: $(TARGETS)
	@for t in $(TARGETS); do\
		echo "Running $$t"; \
		./$$t; \
	done
	

# HCC version:
vadd_hc_arrayview.o: vadd_hc_arrayview.cpp
	$(HCC) $(HCC_CFLAGS) -c $< -o $@
vadd_hc_arrayview: vadd_hc_arrayview.o
	$(HCC) $(HCC_LDFLAGS) $< -o $@


# HCC version, using explicit arrays:
vadd_hc_array.o: vadd_hc_array.cpp
	$(HCC) $(HCC_CFLAGS) -c $< -o $@
vadd_hc_array: vadd_hc_array.o
	$(HCC) $(HCC_LDFLAGS) $< -o $@


# HCC version, using AM (accelerator memory) pointer
vadd_hc_am.o: vadd_hc_am.cpp
	$(HCC) $(HCC_CFLAGS) -c $< -o $@
vadd_hc_am: vadd_hc_am.o
	$(HCC) $(HCC_LDFLAGS) -lhc_am $< -o $@



# HIP version:
vadd_hip.o: vadd_hip.cpp
	$(HIPCC) -c $< -o $@
vadd_hip: vadd_hip.o
	$(HIPCC) $< -o $@


# AMP version:
vadd_amp_arrayview.o: vadd_amp_arrayview.cpp
	$(HCC) $(CPPAMP_CFLAGS) -c $< -o $@
vadd_amp_arrayview: vadd_amp_arrayview.o
	$(HCC) $(CPPAMP_LDFLAGS) $< -o $@
