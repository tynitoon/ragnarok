#include <Windows.h>
#include <stdbool.h>

static bool running;
static BITMAPINFO bitmapInfo;
static void* bitmapMemory;
static HBITMAP bitmapHandle;
static HDC bitmapDeviceContext;

static void Win32ResizeDIBSection(int width, int height)
{
	if (bitmapHandle)
	{
		DeleteObject(bitmapHandle);
	}

	if (!bitmapDeviceContext)
	{
		bitmapDeviceContext = CreateCompatibleDC(0);
	}

	bitmapInfo.bmiHeader.biSize = sizeof(bitmapInfo.bmiHeader);
	bitmapInfo.bmiHeader.biWidth = width;
	bitmapInfo.bmiHeader.biHeight = height;
	bitmapInfo.bmiHeader.biPlanes = 1;
	bitmapInfo.bmiHeader.biBitCount = 32;
	bitmapInfo.bmiHeader.biCompression = BI_RGB;

	bitmapHandle = CreateDIBSection(bitmapDeviceContext, &bitmapInfo, DIB_RGB_COLORS, &bitmapMemory, 0, 0);
}

static void Win32UpdateWindow(HDC deviceContext, int x, int y, int width, int height)
{
	StretchDIBits(deviceContext, x, y, width, height, x, y, width, height, bitmapMemory, &bitmapInfo, DIB_RGB_COLORS, SRCCOPY);
}

LRESULT CALLBACK MainWindowCallBack(HWND window, UINT message, WPARAM wParam, LPARAM lParam)
{
	LRESULT result = 0;

	switch (message)
	{
	case WM_SIZE:
	{
		RECT clientRect;
		GetClientRect(window, &clientRect);
		int width = clientRect.right - clientRect.left;
		int height = clientRect.bottom - clientRect.top;
		Win32ResizeDIBSection(width, height);
	} break;
	case WM_DESTROY:
	{
		running = false;
		OutputDebugString(TEXT("WM_DESTROY\n"));
	} break;
	case WM_CLOSE:
	{
		running = false;
		OutputDebugString(TEXT("WM_CLOSE\n"));
	} break;
	case WM_ACTIVATEAPP:
	{
		OutputDebugString(TEXT("WM_ACTIVATEAPP\n"));
	} break;
	case WM_PAINT:
	{
		PAINTSTRUCT paint;
		HDC deviceContext = BeginPaint(window, &paint);
		int x = paint.rcPaint.left;
		int y = paint.rcPaint.top;
		LONG width = paint.rcPaint.right - paint.rcPaint.left;
		LONG height = paint.rcPaint.bottom - paint.rcPaint.top;
		Win32UpdateWindow(deviceContext, x, y, width, height);
		EndPaint(window, &paint);
	} break;
	default:
	{
		result = DefWindowProc(window, message, wParam, lParam);
		//OutputDebugString(TEXT("default\n"));
	} break;
	}

	return result;
}

int CALLBACK WinMain(HINSTANCE instance, HINSTANCE prevInstance, LPSTR commandLine, int showCode)
{
	WNDCLASS windowClass;

	memset(&windowClass, 0, sizeof(windowClass));
	windowClass.style = CS_OWNDC | CS_HREDRAW | CS_VREDRAW;
	windowClass.lpfnWndProc = MainWindowCallBack;
	windowClass.hInstance = instance;
	//HICON hIcon;
	windowClass.lpszClassName = TEXT("CGodWindowClass");

	if (RegisterClass(&windowClass))
	{
		HWND windowHandle = CreateWindowEx(0, windowClass.lpszClassName, TEXT("CGod"), WS_OVERLAPPEDWINDOW | WS_VISIBLE, CW_USEDEFAULT, CW_USEDEFAULT, CW_USEDEFAULT, CW_USEDEFAULT, 0, 0, instance, 0);
		if (windowHandle)
		{
			running = true;
			while (running)
			{
				MSG message;
				BOOL messageResult = GetMessage(&message, 0, 0, 0);
				if (messageResult > 0)
				{
					TranslateMessage(&message);
					DispatchMessage(&message);
				}
				else
				{
					break;
				}
			}
		}
		else
		{

		}
	}
	else
	{

	}

	return 0;
}