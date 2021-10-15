ELF=main
CC=g++ -std=c++17 -Wall -g
SRC=$(shell find -name '*.cpp' | grep -v .ccls-cache | grep -v hw)
OBJ=$(SRC:.cpp=.o)
$(ELF):$(OBJ)
$(OBJ):

HW2=hw2-2
SRC_HW2=$(shell find -name '*.cpp' | grep -v .ccls-cache | grep -v hw2-3 | grep -v main)
OBJ_HW2=$(SRC_HW2:.cpp=.o)
$(HW2):$(OBJ_HW2)
$(OBJ_HW2):

HW3=hw2-3
SRC_HW3=$(shell find -name '*.cpp' | grep -v .ccls-cache | grep -v hw2-2 | grep -v main)
OBJ_HW3=$(SRC_HW3:.cpp=.o)
$(HW3):$(OBJ_HW3)
$(OBJ_HW3):

.PHONY:clearall clean all
all:$(ELF) $(HW2) $(HW3)
clearall:
	rm -f $(OBJ) $(ELF) $(OBJ_HW2) $(OBJ_HW3) $(HW2) $(HW3)
clean:
	rm -f $(OBJ) $(OBJ_HW2) $(OBJ_HW3)
