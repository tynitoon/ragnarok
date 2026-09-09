/* Exact fused pixel-mode bookkeeping. No game, semantic palette or oracle input.
 * Windows x64 freestanding DLL: no runtime initialization or external imports.
 */
__declspec(dllexport) int reference_update(
    const unsigned int *pixels, unsigned long long n,
    const unsigned short *ids, unsigned int capacity,
    unsigned int *counts, unsigned int *mode, unsigned int *best)
{
    unsigned long long i;
    /* Fail before any mutation if the Python palette must grow. */
    for (i = 0; i < n; ++i) {
        unsigned int colour = pixels[i];
        unsigned int id;
        if (colour >= 16777216u) return -1;
        id = ids[colour];
        if (!id) return 1;
        if (id > capacity) return -2;
    }
    for (i = 0; i < n; ++i) {
        unsigned int colour = pixels[i];
        unsigned long long index = ((unsigned long long)ids[colour] - 1) * n + i;
        unsigned int value = ++counts[index];
        /* Same strict tie rule as the NumPy reference. */
        if (value > best[i]) {
            best[i] = value;
            mode[i] = colour;
        }
    }
    return 0;
}

__declspec(dllexport) void reference_pack(
    const unsigned char *rgb, unsigned long long n, unsigned int *packed)
{
    unsigned long long i;
    for (i = 0; i < n; ++i)
        packed[i] = (unsigned int)rgb[3*i] |
                    ((unsigned int)rgb[3*i+1] << 8) |
                    ((unsigned int)rgb[3*i+2] << 16);
}
