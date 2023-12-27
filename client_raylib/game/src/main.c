#include "raylib.h"
#include "game.h"

t_game_infos game;

static void init_game()
{
	//Init Game
	game.is_running = true;
	game.is_in_game = false;

	//Init Window
	InitWindow(WINDOW_WIDTH, WINDOW_HEIGHT, WINDOW_NAME);
	game.settings.scale = 1.0f;
	SetTargetFPS(60);

	//Init Audio
	InitAudioDevice();
	
	//Init Assets
	init_map(&game.assets);
	load_group_assets(&game.assets, GROUP_MAIN_MENU);
	load_group_assets(&game.assets, GROUP_UI);
}

static void clear_game()
{
	CloseAudioDevice();
	CloseWindow();
}

static void update()
{
	//UpdateMusicStream(music);
	//Basic Handle keys
	if (IsKeyPressed(KEY_ENTER) && (IsKeyDown(KEY_LEFT_ALT) || IsKeyDown(KEY_RIGHT_ALT)))
	{
		if (IsWindowFullscreen())
		{
			ToggleFullscreen();
			SetWindowSize(WINDOW_WIDTH, WINDOW_HEIGHT);
			game.settings.scale = 1.0f;
		}
		else
		{
			int display = GetCurrentMonitor();
			SetWindowSize(GetMonitorWidth(display), GetMonitorHeight(display));
			game.settings.scale = (float)GetMonitorWidth(display) / (float)WINDOW_WIDTH;
			ToggleFullscreen();
		}
	}

	if (game.is_in_game)
	{

	}
	else
	{

	}
}



static void draw()
{
	BeginDrawing();

	//ClearBackground(RAYWHITE); Don't know if it's usefull
	if (game.is_in_game)
	{
	}
	else
	{
		//Display backround
		Texture2D* background = get_map_element(&game.assets, INDEX_MAIN_MENU_BACKGROUND);
		DrawTextureEx(*background, (Vector2){ 0.0f, 0.0f }, 0.0f, game.settings.scale, WHITE);

		//Display centered text
		draw_text("Main Menu", (Vector2) { 960.0f, 30.0f }, 72.0f, MAROON, ALIGNEMENT_CENTER_TOP);
	}

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
		update();
		draw();
	}

	clear_game();

	return 0;
}
