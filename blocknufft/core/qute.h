#ifndef QUTE_H
#define QUTE_H

#include <malloc.h>

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

template <typename T>
class QListPrivate;
template <typename T>
class QList {
public:
	friend class QListPrivate<T>;
	QList();
	virtual ~QList();
	void append(T elmt);
	int count() const;
	T value(int i) const;
private:
	QListPrivate<T> *d;
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


////////// QList implementation -- needs to be in header because template ///////////////
template <typename T>
class QListPrivate {
public:
	QList<T> *q;
	T *m_list;
	int m_internal_size;
	int m_size;
	void allocate(int internal_size);
};

template <typename T>
QList<T>::QList()
{
	d=new QListPrivate<T>;
	d->q=this;
	d->m_internal_size=0;
	d->m_size=0;
	d->m_list=0;
}

template <typename T>
QList<T>::~QList()
{
	if (d->m_list) free(d->m_list);
	delete d;
}

template <typename T>
void QList<T>::append(T elmt)
{
	while (d->m_internal_size<d->m_size+1) d->allocate(d->m_internal_size*2+1);
	d->m_list[d->m_size]=elmt;
	d->m_size++;
}

template <typename T>
int QList<T>::count() const
{
	return d->m_size;
}

template <typename T>
T QList<T>::value(int i) const
{
	if ((i<0)||(i>=d->m_size)) return 0;
	return d->m_list[i];
}

template <typename T>
void QListPrivate<T>::allocate(int internal_size) {
	T *new_list=(T *)malloc(sizeof(T)*internal_size);
	for (int i=0; i<m_size; i++) {
		new_list[i]=m_list[i];
	}
	if (m_list) free(m_list);
	m_list=new_list;
	m_internal_size=internal_size;
}


#endif // QUTE_H

