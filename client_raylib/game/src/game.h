#ifndef GAME_H
#define GAME_H

#include "client.h"
#include "map.h"
#include "assets_loader.h"

#define WINDOW_WIDTH	1920
#define WINDOW_HEIGHT	1080
#define WINDOW_NAME		"Ragnarok Client"

typedef struct		s_game_settings
{
	float			scale;
}					t_game_settings;

typedef struct		s_game_infos
{
	bool			is_running;
	t_server		server;
	t_map			assets;
	t_game_settings	settings;
	bool			is_in_game;
}					t_game_infos;

#endif
