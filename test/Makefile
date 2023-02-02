all: memory

memory:
	$(MAKE) PROJECT_DIR=$(PROJECT_DIR) -C ./memory/

clean:
	$(MAKE) PROJECT_DIR=$(CURRENT_DIR) -C ./memory/ clean

fclean: clean
	$(MAKE) PROJECT_DIR=$(CURRENT_DIR) -C ./memory/ fclean

re: fclean all

.PHONY: all memory clean fclean re
