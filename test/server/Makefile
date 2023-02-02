CC				= gcc

RM				= rm -f

NAME_SERVER		= test_server

NAME_CLIENT		= test_client

SRCS_SERVER		= ./main_server.c				\
				./server.c						\
				../../src/tool/list.c			\
				../../src/tool/single_memory.c

SRCS_CLIENT		= ./main_client.c

OBJS_SERVER		= $(SRCS_SERVER:.c=.o)

OBJS_CLIENT		= $(SRCS_CLIENT:.c=.o)

LDFLAGS			= -L ../../lib/ -lpthread -lsqlite

CFLAGS			= -O3 -Wall -Wextra -Werror -I ../../include/ -I ../../external/

all: test

server: $(OBJS_SERVER)
	$(CC) -o $(NAME_SERVER) $(CFLAGS) $(OBJS_SERVER) $(LDFLAGS)

client: $(OBJS_CLIENT)
	$(CC) -o $(NAME_CLIENT) $(CFLAGS) $(OBJS_CLIENT) $(LDFLAGS)

test: fclean server client
	./$(NAME_SERVER) &
	./$(NAME_CLIENT)

clean:
	$(RM) $(OBJS_SERVER)
	$(RM) $(OBJS_CLIENT)

fclean: clean
	$(RM) $(NAME_SERVER)
	$(RM) $(NAME_CLIENT)
	
re: fclean all

.PHONY: all clean fclean re
