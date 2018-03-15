SRC_DIRS ?= ./src
BUILD_DIR ?= ../../build/shm-cpp-server

SRCS := $(shell find $(SRC_DIRS) -name *.cpp)
OBJS := $(SRCS:%=$(BUILD_DIR)/%.o)
DEPS := $(OBJS:.o=.d)

INC_FLAGS := $(addprefix -I,./include)
CPPFLAGS := $(INC_FLAGS) -MMD -MP --std=c++14
CXXFLAGS := `pkg-config --cflags boost_regex`
LDFLAGS := -lrt

all: server

server: $(OBJS)
	$(CXX) $(OBJS) -o $(BUILD_DIR)/$@ $(LDFLAGS)

$(BUILD_DIR)/%.cpp.o: %.cpp
	$(MKDIR_P) $(dir $@)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -c $< -o $@

format:
	find ./src \( -name *.cpp \) | xargs clang-format -style=file -i

run: $(BUILD_DIR)/server
	$(BUILD_DIR)/server

.PHONY: clean

clean:
	$(RM) -r $(BUILD_DIR)

-include $(DEPS)

MKDIR_P ?= mkdir -p
