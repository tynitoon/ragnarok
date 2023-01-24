CC	= gcc
AR	= ar rcs

RM	= rm -f

NAME	= libsqlite.a

SRCS	= ./sqlite3.c

CFLAGS	=

OBJS	= $(SRCS:.c=.o)

all: server

server: $(OBJS)
	$(AR) $(NAME) $(CFLAGS) $(OBJS)
	mv $(NAME) ../../lib/

clean:
	$(RM) $(OBJS)

fclean: clean
	$(RM) ../../lib/$(NAME)

re: fclean all

.PHONY: all clean fclean re