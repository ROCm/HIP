HIP_PATH?= $(wildcard /opt/rocm/hip)
ifeq (,$(HIP_PATH))
	HIP_PATH=../../..
endif
HIPCC=$(HIP_PATH)/bin/hipcc

CXXFLAGS = -O3

TARGETS=hipBusBandwidth

.PHONY: all clean run install

all: $(TARGETS)

run: $(TARGETS)
	@for t in $(TARGETS); do\
		echo "Running $$t"; \
		./$$t; \
	done

install: $(TARGETS)
	install -t $(HIP_PATH)/bin $(TARGETS)

clean:
	rm -f $(TARGETS) *.o

hipBusBandwidth: hipBusBandwidth.cpp ResultDatabase.cpp
	$(HIPCC) $(CXXFLAGS) $^ -o $@
