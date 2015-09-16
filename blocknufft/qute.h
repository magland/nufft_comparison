#ifndef QUTE_H
#define QUTE_H

class QTimePrivate;
class QTime {
public:
    friend class QTimePrivate;
    QTime();
    virtual ~QTime();
    void start();
    int elapsed();
private:
    QTimePrivate *d;
};

class QStringPrivate;
class QString {
public:
    friend class QStringPrivate;
    QString();
    QString(const char *str);
    virtual ~QString();
private:
    QStringPrivate *d;
};

int qrand();

#endif // QUTE_H

