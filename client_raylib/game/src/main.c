#include "raylib.h"
#include "draw_tool.h"
#include "game.h"

t_game_infos game;

static void init_game()
{
	//Init Game
	game.is_running = true;
	game.is_in_game = false;
	init_list(&game.entities);

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

	//TMP TODO, PROBABLY WILL HAVE TO BE DONE IN SOMETHING LIKE "LoadGameState(MAIN_MENU)" or something like this
	t_list_element* list_element = get_memory(sizeof(t_list_element) + sizeof(t_draw_info) + sizeof(t_draw_texture));
	t_draw_info* info = list_element->data;
	info->type = DRAW_TEXTURE;
	t_draw_texture* texture_info = info->data;
	texture_info->texture = get_map_element(&game.assets, INDEX_MAIN_MENU_BACKGROUND);
	texture_info->src_in_texture.x = 0;
	texture_info->src_in_texture.y = 0;
	texture_info->src_in_texture.width = texture_info->texture->width;
	texture_info->src_in_texture.height = texture_info->texture->height;
	texture_info->position = (Vector2) { 0, 0 };
	texture_info->rotation = 0.0f;
	texture_info->scale = 1.0f;
	texture_info->color = WHITE;
	add_list_element(&game.entities, list_element);

	list_element = get_memory(sizeof(t_list_element) + sizeof(t_draw_info) + sizeof(t_draw_text) + strlen("Main menu"));
	info = list_element->data;
	info->type = DRAW_TEXT;
	t_draw_text* text_info = info->data;
	text_info->mode = ANCHOR_CENTER_TOP;
	text_info->position = (Vector2){ 960.0f, 30.0f };
	text_info->text_size = 72.0f;
	text_info->color = MAROON;
	strcpy(text_info->text, "Main Menu", strlen("Main Menu"));
	add_list_element(&game.entities, list_element);
}

static void clear_game()
{
	CloseAudioDevice();
	CloseWindow();
}

static void update_loop()
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

static void draw_loop()
{
	BeginDrawing();

	ClearBackground(RAYWHITE);
	if (game.is_in_game)
	{
	}
	else
	{
		t_list_element* element = game.entities.head;
		while (element != NULL)
		{
			draw(element->data);
			element = element->next;
		}
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
		update_loop();
		draw_loop();
	}

	clear_game();

	return 0;
}
