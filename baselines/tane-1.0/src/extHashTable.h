#ifndef _H_HASHTABLE
#define _H_HASHTABLE	1


#include "extCandidate.h"
#include "extBitset.h"

#define EXT_MAXHASHSIZE	100000


typedef struct extHashTable {
	int		size;
	int		noofkeys;
	extCandidate	**table;
} extHashTable;

extHashTable *extHashTableNEW(int size);
void extHashTableDESTROY(extHashTable *this);

int extHashTableINSERT(extHashTable *this, uint key, extCandidate *candidate);
int extHashTableEXISTS(extHashTable *this, uint key, extCandidate **candidate);
int extHashTableREMOVE(extHashTable *this, uint key, extCandidate **candidate);

void extHashTablePRINT(extHashTable *this);


#endif
