#ifndef LIST_H
#define LIST_H

typedef struct              s_list_element
{
    struct s_list_element*  prev;
    struct s_list_element*  next;
    char                    buffer[];
}                           t_list_element;

typedef struct              s_list
{
    t_list_element*         head;
    t_list_element*         tail;
}                           t_list;

void add_to_list(t_list* list, char* buffer, size_t size);
void add_list_element_to_list(t_list* list, t_list_element* to_add);
void remove_from_list(t_list* list, t_list_element* to_remove);
void display_list(t_list* list);

#endif