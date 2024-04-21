CXX = aarch64-rockchip-linux-gnu-g++
CFLAGS = -g
LIBS = -lpthread -lasound -lm -lgsl -lgslcblas -lfftw3
TARGETS = doa_locating
OBJS = doa_locating.o pcmAnalysis.o

all: $(TARGETS)

$(TARGETS): $(OBJS)
	$(CXX) $(CFLAGS) -o $@ $^ $(LIBS)

.cpp.o:
	$(CXX) $(CFLAGS) -c $< -o $@

.PHONY : clean
clean:
	rm -f $(TARGETS) *.o
	rm $(TARGET_DIR)/usr/bin/$(TARGETS)

.PHONY : install
install :
	install -D -m 755 $(TARGETS) -t $(TARGET_DIR)/usr/bin/	
	
.PHONY: uninstall
	rm -f $(TARGET_DIR)/usr/bin/$(TARGETS)
