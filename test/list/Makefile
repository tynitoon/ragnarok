CC				= gcc

RM				= rm -f

NAME			= test_list

SRCS			= ./sources/main.c					\
				../../tools/sources/mutex.c			\
				../../tools/sources/single_memory.c \
				../../tools/sources/list.c

OBJS			= $(SRCS:.c=.o)

LDFLAGS			= -lpthread

CFLAGS			= -O3 -Wall -Wextra -Werror -I ../../tools/includes/

all: test

list: $(OBJS)
	$(CC) -o $(NAME) $(CFLAGS) $(OBJS) $(LDFLAGS)

test: fclean list
	./$(NAME)

clean:
	$(RM) $(OBJS)

fclean: clean
	$(RM) $(NAME)
	
re: fclean all

.PHONY: all clean fclean re
