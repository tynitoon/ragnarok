CC				= gcc

RM				= rm -f

NAME			= ragnarok_client

SRCS			= main.c					\
				../tool/single_memory.c

OBJS			= $(SRCS:.c=.o)

LDFLAGS			= -L ../../lib/ -lpthread -lsqlite

#CFLAGS			= -g -ggdb -Wall -Wextra -Werror -I $(PROJECT_DIR)/include/ -I $(PROJECT_DIR)/external/

CFLAGS			= -I ../../include/ -I ../../external/

all: server

server: $(OBJS)
	$(CC) -o $(NAME) $(CFLAGS) $(OBJS) $(LDFLAGS)

clean:
	$(RM) $(OBJS)

fclean: clean
	$(RM) $(NAME)
	
re: fclean all

.PHONY: all clean fclean re
