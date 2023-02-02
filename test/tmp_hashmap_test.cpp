#include <iostream>
#include <string>
#include <unordered_map>
#include <list>

int main()
{

    std::unordered_map<void*, void*> u;

    int j;
    int arr[100000];

    for (j = 0; j < 100000; ++j)
    {
        arr[j] = j;
        u[&arr[j]] = &arr[j];
    }

    while (1);

    //std::list<int> u;

    //int j;
    //int arr[100000];

    //for (j = 0; j < 100000; ++j)
    //{
    //    arr[j] = j;
    //    u.push_back(arr[j]);
    //}
}