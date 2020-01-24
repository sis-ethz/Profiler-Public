#include <stdio.h>
#include <stdlib.h>

#include "extHashTable.h"


static unsigned int prime(unsigned int p);


static unsigned int prime(unsigned int p)
{
	unsigned int q;
	int step = 2;

	if (!(p&1)) p += (step/2);

PRIMETEST:
	q = 3;
	while (q <= p/q) {
		if ((p/q)*q == p) {
			p += step;
			goto PRIMETEST;
		}
		q += 2;
	}

	return(p);
}


extHashTable *extHashTableNEW(int size)
{
	extHashTable *this = NULL;
	extCandidate **table = NULL;
	int i = 0;


	size = prime(size);

	this = (extHashTable *) malloc(sizeof(extHashTable));
	if (this == NULL) {
	   return(NULL);
	}
	this->size = size;
	table = (extCandidate **) malloc(sizeof(extCandidate *)*this->size);
	if (table == NULL) {
	   free(this);
	   return(NULL);
	}
	for (i = 0; i < this->size; i++) table[i] = NULL;

	this->noofkeys = 0;
	this->table = table;

	return(this);
}

void extHashTableDESTROY(extHashTable *this)
{
	free(this->table);
	free(this);
}

int extHashTableINSERT(extHashTable *this, uint name, extCandidate *candidate)
{
	extCandidate *node = NULL;
	int hashkey = 0;

	hashkey = name % this->size;

	node = this->table[hashkey];
	while (node != NULL) {
	      if (node->name == name) return(-100);
	      node = node->next;
	}
	candidate->next = this->table[hashkey];
	this->table[hashkey] = candidate;
	this->noofkeys++;

	return(0);
}

int extHashTableREMOVE(extHashTable *this, uint name, extCandidate **candidate)
{
	extCandidate *node = NULL, *prev = NULL;
	int hashkey = 0;

	hashkey = name % this->size;

	node = this->table[hashkey];
	while (node != NULL) {
		if (node->name == name) {
			*candidate = node;
			if (prev == NULL) this->table[hashkey] = node->next;
			else prev->next = node->next;

			this->noofkeys--;

			return(1);
		}
		prev = node;
		node = node->next;
	}

	return(0);
}

int extHashTableEXISTS(extHashTable *this, uint name, extCandidate **candidate)
{
	extCandidate *node = NULL;
	int hashkey = 0;

	hashkey = name % this->size;

	node = this->table[hashkey];
	while (node != NULL) {
		if (node->name == name) {
			*candidate = node;
			return(1);
		}
		node = node->next;
	}

	return(0);
}

void extHashTablePRINT(extHashTable *this)
{
	int i = 0;
	extCandidate *node = NULL;

	if (this == NULL) return;

	printf("this->size     == %d\n",this->size);
	printf("this->noofkeys == %d\n",this->noofkeys);
	printf("this->table:\n");
	for (i = 0; i < this->size; i++) {
		printf("[%2d]: ",i);
		node = this->table[i];
		while (node != NULL) {
			extBitsetPRINT(node->name); printf("(%d)  ",node->name);
			node = node->next;
		}
		printf("\n");
	}
}
