#include "debug.h"
#include "PVG.h"
#include <cstdlib>


int main(int argc, char *argv[])
{
    PVG pvg("var/waterdrop.pvg");
    pvg.save("resultImg/waterdrop.bmp");

    st_success();
    return EXIT_SUCCESS;
}
