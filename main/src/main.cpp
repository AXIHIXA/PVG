#include "debug.h"
#include "PVGBuffer.h"
#include <cstdlib>


int main(int argc, char *argv[])
{
    PVGBuffer pvg;
    pvg.open("var/egg.pvg", 1);

    st_success();
    return EXIT_SUCCESS;
}
