#ifndef _H_EXTBITSET
#define _H_EXTBITSET	1

#include <sys/types.h>

typedef uint extBitset;


#define extBitsetINSERT(bitset,element) (bitset |= (1 << (element-1)))
#define extBitsetREMOVE(bitset,element) (bitset &= ~(1 << (element-1)))
#define extBitsetUNION(a,b)		(a |= b)
#define extBitsetINTERSECTION(a,b)	(a &= b)
#define extBitsetMINUS(a,b)		(a &= ~b)
#define extBitsetINSERTALL(a,size)	(a = ~(~0 << size))
#define extBitsetCLEAR(a)		(a = 0)
#define extBitsetSETEQUAL(a,b)		(a = b)
#define extBitsetEXISTS(a,element)	(a & (1 << (element-1)))
#define extBitsetISSUPERSET(a,b)	((a & b) != a)
#define extBitsetISSUBSET(a,b)		((a & b) != b)


int extBitsetGETNEXT(uint bitset, int position);
int extBitsetPREFIX(uint a, uint b);

void extBitsetPRINT(uint this);


#endif
