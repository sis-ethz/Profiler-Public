#include <stdio.h>
#include <stdlib.h>

#include "extInputOutput.h"


static uint *tuple_tbl = NULL;


void extInputOutputBEGIN(int nooftuples)
{
	int i = 0;

	tuple_tbl = (uint *) malloc(sizeof(uint)*nooftuples);
	for (i = 0; i < nooftuples; i++) tuple_tbl[i] = 0;
}

void extInputOutputEND(void)
{
	free(tuple_tbl);
}

extPartition *ReadPartition(char *filename)
{
	FILE *fd = NULL;
	extPartition *partition = NULL;
	int number = 0, noofelements = 0, noofsets = 0;


	if ((fd = fopen(filename,"r")) == NULL) return(NULL);

	while (fscanf(fd,"%d\n",&number) == 1) {
	      if (noofsets == 0) noofsets = 1;

	      if (number == 0) {
	         tuple_tbl[noofelements-1] |= endmarker;
	         noofsets++;
	         continue;
	      }

	      tuple_tbl[noofelements++] = number;
	}
	if (noofelements) tuple_tbl[noofelements-1] |= endmarker;

	fclose(fd);


	partition = extPartitionNEW();
	partition->elements = (uint *) malloc(sizeof(uint)*noofelements);
	partition->noofelements = noofelements;
	partition->noofsets = noofsets;
	for (number = 0; number < noofelements; number++) {
	    partition->elements[number] = tuple_tbl[number];
	}

	return(partition);
}
