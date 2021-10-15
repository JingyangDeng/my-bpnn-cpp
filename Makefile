ELF=main
CC=g++ -std=c++17 -Wall -g
SRC=$(shell find -name '*.cpp' | grep -v .ccls-cache)
OBJ=$(SRC:.cpp=.o)
$(ELF):$(OBJ)
$(OBJ):

.PHONY:clean
clean:
	rm -f $(OBJ) $(ELF)
