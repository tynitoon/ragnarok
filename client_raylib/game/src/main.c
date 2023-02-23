#include "raylib.h"
#include "client.h"
#include "map.h"
#include "assets_loader.h"

#define WINDOW_WIDTH	800
#define WINDOW_HEIGHT	450
#define WINDOW_NAME		"Ragnarok Client"

typedef struct  s_game_infos
{
	bool		is_running;
	t_server	server;
	t_map		assets;
}				t_game_infos;

t_game_infos game;

static void init_game()
{
	game.is_running = true;
	InitWindow(WINDOW_WIDTH, WINDOW_HEIGHT, WINDOW_NAME);
	InitAudioDevice();
	init_map(&game.assets);
	SetTargetFPS(60);
	//load_group_assets(&game.assets, GROUP_UI);
}

static void clear_game()
{
	CloseAudioDevice();
	CloseWindow();
}

static void Update()
{
	//UpdateMusicStream(music);
}

static void Draw()
{
	BeginDrawing();

	//ClearBackground(RAYWHITE); Don't know if it's usefull

	EndDrawing();
}

int main(void)
{
	init_game();

	//Set this into each new screen / map / others
	//SetMusicVolume(music, 1.0f);
	//PlayMusicStream(music);

	while (game.is_running)
	{
		Update();
		Draw();
	}

	clear_game();

	return 0;
}
