CURRENT_DIR = $(shell pwd)

all: server client

server:
	$(MAKE) PROJECT_DIR=$(CURRENT_DIR) -C ./src/server/

client:
	$(MAKE) PROJECT_DIR=$(CURRENT_DIR) -C ./src/client/

test:
	$(MAKE) PROJECT_DIR=$(CURRENT_DIR) -C ./test/

clean:
	$(MAKE) PROJECT_DIR=$(CURRENT_DIR) -C ./src/server/ clean
	$(MAKE) PROJECT_DIR=$(CURRENT_DIR) -C ./src/client/ clean
	$(MAKE) PROJECT_DIR=$(CURRENT_DIR) -C ./test/ clean

fclean: clean
	$(MAKE) PROJECT_DIR=$(CURRENT_DIR) -C ./src/server/ fclean
	$(MAKE) PROJECT_DIR=$(CURRENT_DIR) -C ./src/client/ fclean
	$(MAKE) PROJECT_DIR=$(CURRENT_DIR) -C ./test/ fclean

re: fclean all

.PHONY: all test clean fclean re
