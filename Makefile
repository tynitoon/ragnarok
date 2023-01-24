CC	= gcc

RM	= rm -f

NAME	= ragnarok_server

SRCS	= ./src/main.c		\
	  ./src/server.c		\
	  ./src/list.c			\
	  ./src/single_memory.c	\
	  ./src/game.c

CFLAGS	= -g -ggdb -Wall -Wextra -Werror -I include/ -I external/ 

LDFLAGS = -L lib/ -lpthread -lsqlite

#CFLAGS	=  -I include/

OBJS	= $(SRCS:.c=.o)

all: server

server: $(OBJS)
	$(CC) -o $(NAME) $(CFLAGS) $(OBJS) $(LDFLAGS)

clean:
	$(RM) $(OBJS)

fclean: clean
	$(RM) $(NAME)

re: fclean all

.PHONY: all clean fclean re
