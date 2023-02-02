CC				= gcc

RM				= rm -f

NAME			= test_memory

SRCS			= ./main.c						\
				../../src/tool/single_memory.c	\
				../../src/tool/list.c

OBJS			= $(SRCS:.c=.o)

CFLAGS			=  -O3 -Wall -Wextra -Werror -I ../../include/

all: test

memory: $(OBJS)
	$(CC) -o $(NAME) $(CFLAGS) $(OBJS)

test: fclean memory
	./$(NAME)

clean:
	$(RM) $(OBJS)

fclean: clean
	$(RM) $(NAME)
	
re: fclean all

.PHONY: all clean fclean re
