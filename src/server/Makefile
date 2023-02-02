CC				= gcc

RM				= rm -f

NAME			= ragnarok_server

SRCS			= ./main.c					\
				./server.c					\
				./game.c					\
				../tool/list.c				\
				../tool/map.c				\
				../tool/single_memory.c		\
				../tool/hash.c				\
				../tool/sqlite.c

OBJS			= $(SRCS:.c=.o)

LDFLAGS			= -L ../../lib/ -lpthread -lsqlite

CFLAGS			= -g -ggdb -Wall -Wextra -Werror -I ../../include/ -I ../../external/

all: server

server: $(OBJS)
	$(CC) -o $(NAME) $(CFLAGS) $(OBJS) $(LDFLAGS)

clean:
	$(RM) $(OBJS)

fclean: clean
	$(RM) $(NAME)
	
re: fclean all

.PHONY: all clean fclean re
