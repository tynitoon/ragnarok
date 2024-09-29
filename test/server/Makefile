CC				= gcc

RM				= rm -f

NAME_SERVER		= test_server

NAME_CLIENT		= test_client

SRCS_SERVER		= ./sources/main_server.c			\
				./sources/server.c					\
				../../tools/sources/list.c			\
				../../tools/sources/single_memory.c \
				../../tools/sources/mutex.c

SRCS_CLIENT		= ./sources/main_client.c \
				../../tools/sources/single_memory.c \
				../../tools/sources/mutex.c

OBJS_SERVER		= $(SRCS_SERVER:.c=.o)

OBJS_CLIENT		= $(SRCS_CLIENT:.c=.o)

LDFLAGS			= -lpthread

CFLAGS			= -O3 -Wall -Wextra -Werror -I ../../tools/includes -I ../../server/includes/

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
