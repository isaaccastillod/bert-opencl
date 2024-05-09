# CXX := g++
# CC := gcc
# CLANG := clang
# CL := $(CC)
# LLVM_LINK := llvm-link
# LLVM_AS := llvm-as
# LLVM_OPT := opt
# LLVM_DIS := llvm-dis
# CLC := clang
# LDFLAGS := -lOpenCL -lstdc++
# CFLAGS := -Wall -g

# # Source files
# HOST_SOURCE := host.cc
# KERNEL_SOURCE := kernel.cl

# # Executable name
# EXEC := program

# #LLVM kernel  format
# KERNEL_llvm_FILE:=kernel.bc
# KERNEL_SOURCES := kernel.kn

# # Targets
# all: $(EXEC)

# $(EXEC): $(HOST_SOURCE) $(KERNEL_llvm_FILE)
# 	$(CXX) -o $@ $< $(KERNEL_llvm_FILE) $(CFLAGS) $(LDFLAGS)

# $(KERNEL_llvm_FILE): $(KERNEL_SOURCES)
# 	$(CLANG) -x cl -o $(KERNEL_llvm_FILE) -c $(KERNEL_SOURCES)

# clean:
# 	rm -f $(EXEC) $(KERNEL_llvm_FILE) *.o
