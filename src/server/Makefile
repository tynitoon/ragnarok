CC				= gcc

RM				= rm -f

NAME			= $(PROJECT_DIR)/ragnarok_server

SRCS			= $(PROJECT_DIR)/src/server/main.c				\
				$(PROJECT_DIR)/src/server/server.c				\
				$(PROJECT_DIR)/src/server/game.c				\
				$(PROJECT_DIR)/src/tool/list.c					\
				$(PROJECT_DIR)/src/tool/map.c					\
				$(PROJECT_DIR)/src/tool/single_memory.c			\
				$(PROJECT_DIR)/src/tool/hash.c					\
				$(PROJECT_DIR)/src/tool/sqlite.c

OBJS			= $(SRCS:.c=.o)

LDFLAGS			= -L $(PROJECT_DIR)/lib/ -lpthread -lsqlite

CFLAGS			= -g -ggdb -Wall -Wextra -Werror -I $(PROJECT_DIR)/include/ -I $(PROJECT_DIR)/external/

all: server

server: $(OBJS)
	$(CC) -o $(NAME) $(CFLAGS) $(OBJS) $(LDFLAGS)

clean:
	$(RM) $(OBJS)

fclean: clean
	$(RM) $(NAME)
	
re: fclean all

.PHONY: all clean fclean re
