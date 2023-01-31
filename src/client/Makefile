CC				= gcc

RM				= rm -f

NAME			= $(PROJECT_DIR)/ragnarok_client

SRCS			= $(PROJECT_DIR)/src/client/main.c

OBJS			= $(SRCS:.c=.o)

LDFLAGS			= -L $(PROJECT_DIR)/lib/ -lpthread -lsqlite

#CFLAGS			= -g -ggdb -Wall -Wextra -Werror -I $(PROJECT_DIR)/include/ -I $(PROJECT_DIR)/external/

CFLAGS			= -I $(PROJECT_DIR)/include/ -I $(PROJECT_DIR)/external/

all: server

server: $(OBJS)
	$(CC) -o $(NAME) $(CFLAGS) $(OBJS) $(LDFLAGS)

clean:
	$(RM) $(OBJS)

fclean: clean
	$(RM) $(NAME)
	
re: fclean all

.PHONY: all clean fclean re
