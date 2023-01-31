CURRENT_DIR = $(shell pwd)

all: server client

server:
	$(MAKE) PROJECT_DIR=$(CURRENT_DIR) -C ./src/server/

client:
	$(MAKE) PROJECT_DIR=$(CURRENT_DIR) -C ./src/client/

clean:
	$(MAKE) PROJECT_DIR=$(CURRENT_DIR) -C ./src/server/ clean
	$(MAKE) PROJECT_DIR=$(CURRENT_DIR) -C ./src/client/ clean

fclean: clean
	$(MAKE) PROJECT_DIR=$(CURRENT_DIR) -C ./src/server/ fclean
	$(MAKE) PROJECT_DIR=$(CURRENT_DIR) -C ./src/client/ fclean

re: fclean all

.PHONY: all clean fclean re
