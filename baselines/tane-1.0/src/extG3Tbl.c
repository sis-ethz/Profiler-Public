#include <stdio.h>
#include <stdlib.h>
#include <string.h>


#include "extG3Tbl.h"

static int *tuple_tbl = NULL;
static int *tuple_visit = NULL;
static int nooftuples = 0;


void extG3TblINITIALIZE(int argnooftuples)
{
	int i = 0;

	nooftuples = argnooftuples+1;

	tuple_tbl = (int *) malloc(sizeof(int *)*nooftuples);
	for (i = 0; i < nooftuples; i++) tuple_tbl[i] = 0;

	tuple_visit = (uint *) malloc(sizeof(uint *)*nooftuples);
	for (i = 0; i < nooftuples; i++) tuple_visit[i] = 0;
}

void extG3TblLOAD(extPartition *partition)
{
	int i = 0, size = 0;

	while (i < partition->noofelements) {
	    size = 1;
	    while (1) {
	        if (partition->elements[i++] & endmarker) break;
	        size++;
	    }
	    tuple_tbl[(int) (partition->elements[i-1] & ~endmarker)-1] = size;
	}
	/*
	printf("tuple_tbl:\n");
	for (i = 0; i < nooftuples; i++) {
		printf("[%2d] = %d\n",i,tuple_tbl[i]);
	}
	*/
}

void extG3TblUNLOAD(extPartition *partition)
{
	int i = 0;

	for (i = 0; i < partition->noofelements; i++)
		tuple_tbl[(partition->elements[i] & ~endmarker)-1] = 0;
}


int G3(extPartition *b)
{
	int i = 0, elements = 0, biggest = 0, sum = 0;


	while (i < b->noofelements) {
		biggest = 0;
		while ((b->elements[i++] & endmarker) == 0) {
			//printf("b->elements[%d] == %d\n",i-1,b->elements[i-1]);
			elements = tuple_tbl[b->elements[i-1]-1];
			if (elements > biggest) biggest = elements;
			//printf("elements == %d\n",elements);
		}
		//printf("b->elements[%d] == %d\n",i-1,b->elements[i-1] & ~endmarker);
		elements = tuple_tbl[(b->elements[i-1] & ~endmarker)-1];
		//printf("elements == %d\n",elements);
		if (elements > biggest) biggest = elements;

		sum += biggest;
	}
	//printf("sum == %d\n",sum);


	return(b->noofelements-sum);
}

int extG3BiggestSet(extPartition *partition)
{
	int i = 0, size = 0, biggest = 0;

	while (i < partition->noofelements) {
	    size = 1;
	    while (1) {
	        if (partition->elements[i++] & endmarker) break;
	        size++;
	    }
	    if (size > biggest) biggest = size;
	}

	return(biggest);
}
