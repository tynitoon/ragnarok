CC				= gcc

RM				= rm -f

NAME			= test_01

SRCS			= ./main.c		\
				../../src/tool/single_memory.c \
				../../src/tool/list.c

OBJS			= $(SRCS:.c=.o)

CFLAGS			= -O3 -I ../../include/

all: memory

memory: $(OBJS)
	$(CC) -o $(NAME) $(CFLAGS) $(OBJS)
	./test_01

clean:
	$(RM) $(OBJS)

fclean: clean
	$(RM) $(NAME)
	
re: fclean all

.PHONY: all clean fclean re
