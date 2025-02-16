#include "test.h"

#include <windows.h>
#include <stdio.h>

#include "single_memory.h"
#include "message.h"

DWORD WINAPI thread_function(LPVOID lpParam)
{
	t_game *game = lpParam;
	
	while (1)
	{
		t_message *msg;
		while ((msg = list_remove_front(&game->messages_received)) != NULL)
		{
			printf("message received : %s\n", msg->buffer);
			FREE(msg);
			t_message *new_msg = MALLOC(sizeof(t_message) + strlen("ta mere") + 1);
			new_msg->size = sizeof(t_message) + strlen("ta mere") + 1;
			strcpy(new_msg->buffer, "ta mere");
			list_add_back(&game->messages_to_send, new_msg);
		}
	}
	return 0;
}

void start_thread(t_game *game)
{
	t_message* new_msg = MALLOC(sizeof(t_message) + strlen("First Message"));
	new_msg->size = sizeof(t_message) + strlen("First Message");
	strcpy(new_msg->buffer, "First Message");
	list_add_back(&game->messages_to_send, new_msg);

	// Création du thread
	HANDLE thread_handle = CreateThread(
		NULL, // Attributs de sécurité (par défaut)
		0, // Taille de la pile (par défaut)
		thread_function, // Fonction du thread
		game, // Paramètre du thread
		0, // Flags de création (par défaut)
		NULL // ID du thread (pas utilisé ici)
	);

	// Vérification de la création du thread
	if (thread_handle == NULL) {
		fprintf(stderr, "Erreur lors de la création du thread : %d\n", GetLastError());
		return;
	}

	// Détachement du thread
	if (!CloseHandle(thread_handle)) {
		fprintf(stderr, "Erreur lors du détachement du thread : %d\n", GetLastError());
		return;
	}

	// Le thread continue de s'exécuter en arrière-plan
	printf("Thread détaché. Le programme principal continue...\n");
}