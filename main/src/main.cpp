#include "debug.h"
#include "PVG.h"
#include <cstdlib>


int main(int argc, char *argv[])
{
    PVG pvg("var/apple.pvg");
    pvg.save("resultImg/apple.bmp");

    st_success();
    return EXIT_SUCCESS;
}
