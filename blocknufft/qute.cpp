#include "qute.h"
#include <math.h>
#include <stdlib.h>
#include <time.h>

using namespace std;

//typedef chrono::high_resolution_clock Clock;
class QTimePrivate {
public:
    QTime *q;

	time_t m_start_time;

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
	d->m_start_time=clock();
}

int QTime::elapsed()
{
	time_t t2 = clock();
	return t2-d->m_start_time;
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
