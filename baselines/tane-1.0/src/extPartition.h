#ifndef _H_EXTPARTITION
#define _H_EXTPARTITION	1

#include <sys/types.h>


typedef struct extPartition {
	int	noofsets;
	int	noofelements;
#ifdef USEDISK
	int	position;
#endif
	uint	*elements;
} extPartition;


extPartition *extPartitionNEW(void);
void extPartitionDESTROY(extPartition *this);

void extPartitionPRINT(extPartition *this);

#define endmarker	((uint)1 << 31)


#endif
