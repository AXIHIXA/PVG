#include "debug.h"
#include "PVG.h"
#include <cstdlib>


int main(int argc, char *argv[])
{
    QString test = argv[1];

    QElapsedTimer timer;
    timer.start();

    PVG pvg(QString("var/%1.pvg").arg(test));
    pvg.save(QString("resultImg/%1.bmp").arg(test));

    st_info("cost %lld s", timer.elapsed() / 1000);

    st_success();
    return EXIT_SUCCESS;
}
