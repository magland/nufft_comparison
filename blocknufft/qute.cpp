#include "qute.h"
#include <math.h>
#include <stdlib.h>
#include <time.h>

using namespace std;

//typedef chrono::high_resolution_clock Clock;
class QTimePrivate {
public:
    QTime *q;

    struct timespec m_start_time;

};

QTime::QTime() {
    d=new QTimePrivate;
    d->q=this;
}

QTime::~QTime()
{
    delete d;
}

void QTime::start()
{
    clock_gettime(CLOCK_MONOTONIC, &d->m_start_time);
}

int QTime::elapsed()
{
    struct timespec t2;
    clock_gettime(CLOCK_MONOTONIC, &t2);
    double elapsed;
    elapsed = (t2.tv_sec - d->m_start_time.tv_sec);
    elapsed += (t2.tv_nsec - d->m_start_time.tv_nsec) / 1000000000.0;
    return (int)(elapsed*1000);
}

int qrand()
{
    return rand();
}

class QStringPrivate {
public:
    QString *q;
};

QString::QString() {
    d=new QStringPrivate;
    d->q=this;
}

QString::~QString()
{
    delete d;
}
