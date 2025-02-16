CC				= gcc

RM				= rm -f

NAME			= ragnarok_server

SRCS			= ./sources/main.c					\
				./sources/server.c					\
				./sources/game.c					\
				../tools/sources/list.c				\
				../tools/sources/map.c				\
				../tools/sources/single_memory.c	\
				../tools/sources/hash.c				\
				../tools/sources/mutex.c			\
				../tools/sources/sqlite.c

OBJS			= $(SRCS:.c=.o)

LDFLAGS			= -L ../lib/ -lpthread -lsqlite

CFLAGS			= -g -ggdb -Wall -Wextra -Werror -I ./includes -I ../tools/includes/ -I ../externals/sqlite/includes/

all: server

server: $(OBJS)
	$(CC) -o $(NAME) $(CFLAGS) $(OBJS) $(LDFLAGS)

clean:
	$(RM) $(OBJS)

fclean: clean
	$(RM) $(NAME)
	
re: fclean all

.PHONY: all clean fclean re
