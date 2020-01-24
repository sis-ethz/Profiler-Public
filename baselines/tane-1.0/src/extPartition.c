#include <stdio.h>
#include <stdlib.h>

#include "extPartition.h"


extPartition *extPartitionNEW(void)
{
	extPartition *this = NULL;

	if ((this = (extPartition *) malloc(sizeof(extPartition))) == NULL) {
	   return(NULL);
	}

	this->noofsets = 0;
	this->noofelements = 0;
	this->elements = NULL;
#ifdef USEDISK
	this->position = -1;
#endif

	return(this);
}

void extPartitionDESTROY(extPartition *this)
{
	free(this->elements);
	free(this);
}

void extPartitionPRINT(extPartition *this)
{
	int i = 0;

	for (i = 0; i < this->noofelements; i++) {
	    printf("%d ",this->elements[i] & ~endmarker);
	    if (this->elements[i] & endmarker) printf("\n");
	}
}
