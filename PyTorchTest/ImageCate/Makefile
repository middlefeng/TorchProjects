

TORCH_PATH      := ../../../../lib/python3.12/site-packages/torch/include/torch/csrc/api/include
TORCH_CSRC_PATH := ../../../../lib/python3.12/site-packages/torch/include

TORCH_LIB_PATH  := ../../../../lib64/python3.12/site-packages/torch/lib

FLAGS := -I$(TORCH_PATH) -I. -I$(TORCH_CSRC_PATH) -L$(TORCH_LIB_PATH) -std=c++17 -D_GLIBCXX_USE_CXX11_ABI=0 -g -rdynamic
LIBS  := -ltorch -lc10 -ltorch_cpu -Wl,-rpath,$(TORCH_LIB_PATH)


all: main

main: main.o ImageSet.o
	g++ $^ $(FLAGS) $(LIBS) -o main

%.o: %.cpp
	g++ $(FLAGS) -c $< -o $@


.PHONY: all
	g++ 
