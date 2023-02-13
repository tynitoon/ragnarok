#include <sys/time.h>
#include <string.h>

#include "game.h"
#include "server.h"

int main()
{
	t_game_infos	game_infos;

	memset(&game_infos, 0, sizeof(t_game_infos));

	start_server(4242, &game_infos.clients);

	return 0;
}