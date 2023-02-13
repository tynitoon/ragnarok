CC				= g++

RM				= rm -f

NAME			= test_map

SRCS			= ./sources/main.c					\
				../../tools/sources/single_memory.c	\
				../../tools/sources/map.c

OBJS			= $(SRCS:.c=.o)

LDFLAGS			= -lpthread

CFLAGS			= -O3 -Wall -Wextra -Werror -I ../../tools/includes/

all: test

map: $(OBJS)
	$(CC) -o $(NAME) $(CFLAGS) $(OBJS) $(LDFLAGS)

test: fclean map
	./$(NAME)

clean:
	$(RM) $(OBJS)

fclean: clean
	$(RM) $(NAME)
	
re: fclean all

.PHONY: all clean fclean re
