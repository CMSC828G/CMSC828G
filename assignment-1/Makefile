# NOTE: Do NOT modify this file

CXX := mpicxx
CXXFLAGS := -O2 -std=c++17 -Wall -Wextra -Wpedantic

TARGET := harness
TARGET_CORRECT := harness_correct
SRCS := harness.cc collectives.cc

# For reference implementation
SRCS_CORRECT := harness.cc collectives_correct.cc
OBJS := $(SRCS:.cc=.o)

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CXX) $(CXXFLAGS) -o $@ $^

$(TARGET_CORRECT): $(SRCS_CORRECT:.cc=.o)
	$(CXX) $(CXXFLAGS) -o $@ $^

%.o: %.cc
	$(CXX) $(CXXFLAGS) -c $< -o $@

clean:
	rm -f $(OBJS) $(SRCS_CORRECT:.cc=.o) $(TARGET) $(TARGET_CORRECT)

.PHONY: all clean
